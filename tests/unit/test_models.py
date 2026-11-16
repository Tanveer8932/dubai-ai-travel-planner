"""AC1.1: models round-trip through JSON; invalid data raises ValidationError."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import pytest
from factories import AS_OF, itinerary, price, provenance, slot, trip_request
from pydantic import BaseModel, ValidationError

import dubai_planner.models as m
from dubai_planner.data.seed import load_catalog
from dubai_planner.models import (
    Activity,
    ActivityKind,
    BudgetCategory,
    BudgetPlan,
    CategoryAllocation,
    ChatMessage,
    ChatRole,
    CostBreakdown,
    CostItem,
    DailyWeather,
    DataKind,
    DayPlan,
    EventCategory,
    EventInfo,
    GeoPoint,
    Itinerary,
    OpeningHours,
    PriceRange,
    PriceUnit,
    Provenance,
    ResearchBundle,
    ReviewIssue,
    ReviewReport,
    RuleViolation,
    Severity,
    TraceEvent,
    TraceEventType,
    TransportLeg,
    TransportMode,
    TransportPlan,
    Travellers,
    TripRequest,
    TripRequestDraft,
    WeatherContext,
    WeatherSource,
    Weekday,
)
from dubai_planner.models.budget import BudgetScope


def _weather() -> WeatherContext:
    return WeatherContext(
        source=WeatherSource.FORECAST,
        days=[
            DailyWeather(
                date=date(2027, 7, 1),
                temp_max_c=42.5,
                temp_min_c=31.0,
                precipitation_mm=0,
                uv_index_max=11,
            )
        ],
        provenance=provenance(DataKind.API),
    )


def _samples() -> list[BaseModel]:
    catalog = load_catalog()
    return [
        trip_request(),
        Travellers(adults=2, children_ages=[6, 9]),
        catalog.attractions[0],
        catalog.hotels[0],
        catalog.restaurants[0],
        TransportLeg(
            mode=TransportMode.TAXI, distance_km=12.3, duration_min=25, cost_aed=Decimal("38.60")
        ),
        TransportPlan(
            primary_modes=[TransportMode.METRO, TransportMode.TAXI],
            estimated_daily_cost_aed=Decimal("40"),
            estimated_total_cost_aed=Decimal("160"),
            provenance=provenance(DataKind.ESTIMATED),
        ),
        _weather(),
        EventInfo(
            id="evt_x_2027",
            name="X",
            category=EventCategory.FESTIVAL,
            start_date=date(2027, 1, 1),
            end_date=date(2027, 1, 5),
            approximate=True,
            provenance=provenance(DataKind.ESTIMATED),
        ),
        BudgetPlan(
            budget_aed=Decimal("8000"),
            allocations=[
                CategoryAllocation(category=BudgetCategory.FOOD, amount_aed=Decimal("1200.50"))
            ],
        ),
        itinerary().cost,
        slot("09:00", "10:00"),
        itinerary().days[0],
        itinerary(),
        RuleViolation(
            rule_id="heat.midday_outdoor",
            severity=Severity.ERROR,
            message="too hot",
            day_index=0,
            slot_index=2,
            details={"temp": 42},
        ),
        ReviewIssue(severity=Severity.WARNING, message="far", suggested_fix="swap"),
        ReviewReport(
            approved=False,
            score=0.4,
            issues=[ReviewIssue(severity=Severity.ERROR, message="over budget")],
        ),
        TraceEvent(
            run_id="r1",
            mode="langgraph",
            agent="budget",
            event=TraceEventType.END,
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            ended_at=datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC),
            tokens_in=10,
            tokens_out=5,
            payload={"k": "v"},
        ),
        ChatMessage(role=ChatRole.USER, content="4 days in March, AED 3000"),
        ResearchBundle(attractions=catalog.attractions[:2], weather=_weather()),
        TripRequestDraft(budget_aed=Decimal("5000")),
    ]


@pytest.mark.parametrize("obj", _samples(), ids=lambda o: type(o).__name__)
def test_round_trip_json(obj: BaseModel) -> None:
    restored = type(obj).model_validate_json(obj.model_dump_json())
    assert restored == obj


def test_every_exported_model_has_json_schema() -> None:
    """Needed for LLM structured output (LangChain) and CrewAI ``output_pydantic``."""
    for name in m.__all__:
        obj = getattr(m, name)
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            schema = obj.model_json_schema()
            assert schema["type"] == "object", name


# ----------------------------------------------------------------------------- invalid data


def test_end_before_start_rejected() -> None:
    with pytest.raises(ValidationError, match="end_date must be on or after start_date"):
        trip_request(start_date=date(2027, 2, 12), end_date=date(2027, 2, 10))


def test_trip_too_long_rejected() -> None:
    with pytest.raises(ValidationError, match="longer than 30 days"):
        trip_request(end_date=date(2027, 2, 10) + timedelta(days=30))


@pytest.mark.parametrize(
    "build",
    [
        lambda: slot("09:00", "10:00", cost="-1"),
        lambda: price(lo="-5", hi="10"),
        lambda: CostItem(category=BudgetCategory.FOOD, description="x", amount_aed=-1),
        lambda: TransportLeg(
            mode=TransportMode.WALK, distance_km=1, duration_min=5, cost_aed=Decimal("-0.01")
        ),
        lambda: CostBreakdown(total=Decimal("-10")),
        lambda: CategoryAllocation(category=BudgetCategory.FOOD, amount_aed=Decimal("-1")),
        lambda: trip_request(budget_aed=Decimal("0")),
    ],
    ids=["slot", "price", "cost_item", "leg", "breakdown", "allocation", "zero_budget"],
)
def test_negative_costs_rejected(build: Any) -> None:
    with pytest.raises(ValidationError):
        build()


def test_overlapping_slots_rejected() -> None:
    with pytest.raises(ValidationError, match="overlapping slots"):
        DayPlan(date=date(2027, 2, 10), slots=[slot("09:00", "11:00"), slot("10:30", "12:00")])


def test_unordered_slots_rejected() -> None:
    with pytest.raises(ValidationError, match="start-time order"):
        DayPlan(date=date(2027, 2, 10), slots=[slot("14:00", "15:00"), slot("09:00", "10:00")])


def test_adjacent_slots_allowed() -> None:
    day = DayPlan(date=date(2027, 2, 10), slots=[slot("09:00", "10:00"), slot("10:00", "11:00")])
    assert len(day.slots) == 2


def test_slot_end_must_follow_start() -> None:
    with pytest.raises(ValidationError, match="must be after start"):
        slot("10:00", "10:00")


def test_venue_activity_requires_ref_id() -> None:
    """Grounding at the contract level: venue activities must reference an id."""
    with pytest.raises(ValidationError, match="must reference a venue ref_id"):
        Activity(kind=ActivityKind.ATTRACTION, name="Invented Tower")
    assert Activity(kind=ActivityKind.FREE_TIME, name="Rest").ref_id is None


def test_itinerary_days_unique_and_ordered() -> None:
    it = itinerary()
    with pytest.raises(ValidationError, match="unique dates"):
        Itinerary.model_validate({**it.model_dump(), "days": [it.days[1], it.days[0]]})
    with pytest.raises(ValidationError):
        Itinerary.model_validate({**it.model_dump(), "days": []})


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TripRequest.model_validate({**trip_request().model_dump(), "flights": True})


def test_child_ages_validated() -> None:
    with pytest.raises(ValidationError):
        Travellers(adults=1, children_ages=[18])
    with pytest.raises(ValidationError):
        Travellers(adults=0)


def test_weather_and_events_validated() -> None:
    with pytest.raises(ValidationError, match="temp_min_c"):
        DailyWeather(date=date(2027, 1, 1), temp_max_c=20, temp_min_c=25)
    with pytest.raises(ValidationError, match="event end_date"):
        EventInfo(
            id="e",
            name="e",
            category=EventCategory.FESTIVAL,
            start_date=date(2027, 1, 2),
            end_date=date(2027, 1, 1),
            provenance=provenance(),
        )


def test_trace_event_end_not_before_start() -> None:
    with pytest.raises(ValidationError):
        TraceEvent(
            run_id="r",
            mode="m",
            agent="a",
            event=TraceEventType.END,
            started_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            ended_at=datetime(2026, 1, 1, 0, tzinfo=UTC),
        )


def test_review_and_violation_validation() -> None:
    with pytest.raises(ValidationError):
        ReviewReport(approved=True, score=1.5)
    with pytest.raises(ValidationError):
        RuleViolation(rule_id="not a rule id", severity=Severity.INFO, message="x")


# ----------------------------------------------------------------------------- money & provenance


def test_money_is_exact_decimal_with_two_places() -> None:
    p = PriceRange(
        min_aed=0.1,
        max_aed="12.345",
        unit=PriceUnit.PER_PERSON,  # type: ignore[arg-type]
        provenance=provenance(DataKind.ESTIMATED),
    )
    assert p.min_aed == Decimal("0.10")
    assert p.max_aed == Decimal("12.35")  # half-up
    assert isinstance(p.max_aed, Decimal)
    with pytest.raises(ValidationError):
        PriceRange(
            min_aed="NaN",
            max_aed=1,
            unit=PriceUnit.PER_PERSON,  # type: ignore[arg-type]
            provenance=provenance(),
        )


def test_price_range_rules_and_labels() -> None:
    with pytest.raises(ValidationError, match="min_aed"):
        price(lo="30", hi="20")
    with pytest.raises(ValidationError, match="FREE"):
        price(lo="0", hi="5", unit=PriceUnit.FREE)
    est = price("150", "300")
    assert est.is_estimate and est.label().startswith("≈ AED 150–300 (estimate")
    assert est.midpoint_aed == Decimal("225.00")
    verified = est.model_copy(
        update={"provenance": provenance(DataKind.VERIFIED, source_url="https://example.org/fares")}
    )
    assert not verified.is_estimate and verified.label() == "AED 150–300"
    assert price("0", "0").label() == "Free"


def test_verified_provenance_requires_url() -> None:
    with pytest.raises(ValidationError, match="source_url"):
        Provenance(kind=DataKind.VERIFIED, source="x", as_of=AS_OF)
    p = Provenance(kind=DataKind.ESTIMATED, source="seed", as_of=AS_OF)
    assert p.is_estimate and p.label() == "estimated · seed · as of 2026-11-15"


def test_geo_distance() -> None:
    frame = GeoPoint(lat=25.2354, lon=55.3004)
    burj = GeoPoint(lat=25.1972, lon=55.2744)
    assert 4.5 < frame.distance_km(burj) < 5.2
    assert frame.distance_km(frame) == 0


def test_opening_hours() -> None:
    mall = OpeningHours(open=time(10), close=time(0))
    assert mall.past_midnight and mall.is_open(Weekday.MON, time(22), time(23, 30))
    museum = OpeningHours(open=time(10), close=time(18), closed_days=[Weekday.MON])
    assert museum.is_open(Weekday.TUE, time(10), time(12))
    assert not museum.is_open(Weekday.MON, time(10), time(12))
    assert not museum.is_open(Weekday.TUE, time(17), time(19))


# ----------------------------------------------------------------------------- request & budget


def test_total_budget_by_scope() -> None:
    req = trip_request(budget_aed=Decimal("1000"))  # 3 days, 3 travellers
    assert req.num_days == 3 and req.nights == 2 and req.travellers.total == 3
    assert req.total_budget_aed() == Decimal("1000.00")
    assert req.model_copy(
        update={"budget_scope": BudgetScope.PER_PERSON}
    ).total_budget_aed() == Decimal("3000.00")
    assert req.model_copy(
        update={"budget_scope": BudgetScope.PER_DAY}
    ).total_budget_aed() == Decimal("3000.00")


def test_draft_missing_fields_and_promotion() -> None:
    draft = TripRequestDraft(interests=["beach"])
    assert draft.missing_fields() == ["start_date", "end_date", "budget_aed"]
    assert not draft.is_complete
    with pytest.raises(ValueError, match="missing: start_date, end_date, budget_aed"):
        draft.to_request()
    full = draft.model_copy(
        update={
            "start_date": date(2027, 2, 1),
            "end_date": date(2027, 2, 5),
            "budget_aed": Decimal("8000"),
        }
    )
    req = full.to_request()
    assert isinstance(req, TripRequest) and req.interests == ["beach"]


def test_cost_breakdown_from_items_exact_arithmetic() -> None:
    items = [
        CostItem(
            category=BudgetCategory.ACCOMMODATION, description="a", amount_aed=Decimal("0.10")
        ),
        CostItem(
            category=BudgetCategory.ACCOMMODATION, description="b", amount_aed=Decimal("0.20")
        ),
        CostItem(category=BudgetCategory.FOOD, description="c", amount_aed=Decimal("100.05")),
    ]
    cb = CostBreakdown.from_items(items, budget_aed=Decimal("100"), tolerance=Decimal("0.05"))
    assert cb.accommodation == Decimal("0.30")  # not 0.30000000000000004
    assert cb.total == Decimal("100.35") == cb.category_sum
    assert cb.is_consistent and cb.within_budget and cb.is_estimate
    assert cb.overshoot_aed == Decimal("0.35")
    over = CostBreakdown.from_items(items, budget_aed=Decimal("95"))
    assert not over.within_budget  # 100.35 > 95 * 1.05 = 99.75
    broken = cb.model_copy(update={"total": Decimal("1")})
    assert not broken.is_consistent  # representable, so the validator rule can report it


def test_budget_plan_helpers_and_rules() -> None:
    plan = BudgetPlan(
        budget_aed=Decimal("1000"),
        allocations=[
            CategoryAllocation(category=BudgetCategory.ACCOMMODATION, amount_aed=Decimal("600")),
            CategoryAllocation(category=BudgetCategory.FOOD, amount_aed=Decimal("450")),
        ],
    )
    assert plan.allocation(BudgetCategory.FOOD) == Decimal("450.00")
    assert plan.allocation(BudgetCategory.TRANSPORT) == Decimal("0")
    assert plan.allocated_total_aed == Decimal("1050.00")
    assert plan.unallocated_aed == Decimal("-50.00")
    with pytest.raises(ValidationError, match="only once"):
        BudgetPlan(
            budget_aed=Decimal("1"),
            allocations=[
                CategoryAllocation(category=BudgetCategory.FOOD, amount_aed=1),  # type: ignore[arg-type]
                CategoryAllocation(category=BudgetCategory.FOOD, amount_aed=2),  # type: ignore[arg-type]
            ],
        )
    with pytest.raises(ValidationError, match="explain why"):
        BudgetPlan(budget_aed=Decimal("1"), allocations=[], feasible=False)


def test_itinerary_helpers() -> None:
    it = itinerary(days=3)
    assert it.start_date == date(2027, 2, 10) and it.end_date == date(2027, 2, 12)
    assert it.venue_ids() == {
        "att_dubai_frame",
        "res_xva_cafe",
        "att_dubai_fountain",
        "htl_rove_downtown",
    }
    assert it.days[0].slot_cost_sum == Decimal("110.00")
    assert it.days[0].slots[0].duration_min == 90


def test_research_bundle_known_ids() -> None:
    catalog = load_catalog()
    bundle = ResearchBundle(attractions=catalog.attractions[:2], hotels=catalog.hotels[:1])
    assert bundle.known_ids() == {
        catalog.attractions[0].id,
        catalog.attractions[1].id,
        catalog.hotels[0].id,
    }


def test_chat_message_length_limit() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role=ChatRole.USER, content="x" * 20_001)


def test_weather_helpers() -> None:
    w = _weather()
    assert w.max_temp_c == 42.5 and w.hot_days == [date(2027, 7, 1)]
    assert w.for_date(date(2027, 7, 1)) is not None and w.for_date(date(2027, 7, 2)) is None
