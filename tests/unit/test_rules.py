"""AC2.5: every deterministic rule has a positive (fires) and a negative (silent) case."""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal
from typing import Any

import pytest
from factories import trip_request

from dubai_planner.models import (
    Activity,
    ActivityKind,
    DailyWeather,
    DataKind,
    DayPlan,
    Itinerary,
    PlannerMode,
    Provenance,
    ResearchBundle,
    RuleViolation,
    Severity,
    TimeSlot,
    TransportLeg,
    TransportMode,
    Travellers,
    TripRequest,
    WeatherContext,
    WeatherSource,
)
from dubai_planner.tools.costs import calculate_costs, day_cost, party_cost
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.validation.rules import RULES, has_errors, validate_itinerary

FEB = date(2027, 2, 10)  # mild
JULY = date(2027, 7, 14)  # hot (climatology > 38 °C)
OCT = date(2027, 10, 20)  # outside Ramadan; not a "hot month"


def _slot(tb: Toolbox, ref: str, start: str, end: str, *, kind: ActivityKind | None = None,
          leg_min: int | None = None, indoor: bool | None = None,
          party: Travellers | None = None) -> TimeSlot:
    venue = tb.catalog.venue(ref)
    party = party or Travellers()
    if venue is None:  # ungrounded/fake venue
        k = kind or ActivityKind.ATTRACTION
        return TimeSlot(start=time.fromisoformat(start), end=time.fromisoformat(end),
                        activity=Activity(kind=k, name=ref, ref_id=ref),
                        indoor=True if indoor is None else indoor)
    if ref.startswith("res_"):
        k, cost, ind = ActivityKind.DINING, party_cost(venue.avg_spend, party), True  # type: ignore[union-attr]
    else:
        k, cost, ind = ActivityKind.ATTRACTION, party_cost(venue.price, party), venue.indoor  # type: ignore[union-attr]
    leg = None
    if leg_min is not None:
        leg = TransportLeg(mode=TransportMode.TAXI, distance_km=50, duration_min=leg_min,
                           cost_aed=Decimal("80"), to_ref=ref)
    return TimeSlot(start=time.fromisoformat(start), end=time.fromisoformat(end),
                    activity=Activity(kind=kind or k, name=venue.name, ref_id=ref,
                                      area=venue.area, location=venue.location),
                    transport_from_prev=leg, cost_aed=cost,
                    indoor=ind if indoor is None else indoor)


def _itinerary(tb: Toolbox, slots: list[TimeSlot], *, day: date = FEB,
               request: TripRequest | None = None) -> Itinerary:
    req = request or _request(day)
    plan = DayPlan.model_construct(date=day, theme="", slots=slots, day_cost_aed=Decimal(0),
                                   notes=[])  # model_construct: allow broken days for rules
    plan = plan.model_copy(update={"day_cost_aed": day_cost(plan)})
    cost = calculate_costs([plan], hotel=None, nights=0, travellers=req.travellers,
                           budget_aed=req.total_budget_aed(), contingency_pct=Decimal("0.1"))
    base = Itinerary(session_id="s", mode=PlannerMode.LANGGRAPH, days=[DayPlan(date=day)],
                     cost=cost)
    return base.model_copy(update={"days": [plan]})  # no re-validation of the broken day


def _request(day: date = FEB, **kw: Any) -> TripRequest:
    data: dict[str, Any] = {"start_date": day, "end_date": day, "travellers": Travellers(),
                            "budget_aed": Decimal("5000"), "dietary": []}
    data.update(kw)
    return trip_request(**data)


def _fired(violations: list[RuleViolation], rule: str) -> list[RuleViolation]:
    return [v for v in violations if v.rule_id == rule]


def _check(tb: Toolbox, it: Itinerary, req: TripRequest, research: ResearchBundle | None = None,
           ) -> list[RuleViolation]:
    return validate_itinerary(it, req, research, catalog=tb.catalog, tolerance=Decimal("0.05"))


def _clean_day(tb: Toolbox) -> list[TimeSlot]:
    return [_slot(tb, "att_dubai_frame", "09:00", "10:30"),
            _slot(tb, "res_xva_cafe", "12:30", "13:30")]


def test_all_eleven_rules_registered() -> None:
    assert set(RULES) == {
        "budget.total", "budget.arithmetic", "time.overlap", "time.opening_hours",
        "heat.midday_outdoor", "geo.travel_time", "pace.density", "constraints.dietary",
        "constraints.accessibility", "culture.ramadan", "grounding.known_entities"}


def test_clean_itinerary_has_no_violations(toolbox: Toolbox) -> None:
    req = _request(OCT)
    assert _check(toolbox, _itinerary(toolbox, _clean_day(toolbox), day=OCT, request=req),
                  req) == []


# ------------------------------------------------------------------ budget.total


def test_budget_total(toolbox: Toolbox) -> None:
    it = _itinerary(toolbox, _clean_day(toolbox))
    assert not _fired(_check(toolbox, it, _request()), "budget.total")
    tight = _request(budget_aed=it.cost.total / Decimal("1.06"))
    [v] = _fired(_check(toolbox, it, tight), "budget.total")
    assert v.severity is Severity.ERROR and "exceeds" in v.message
    within_tolerance = _request(budget_aed=(it.cost.total / Decimal("1.04")).quantize(
        Decimal("0.01")))
    assert not _fired(_check(toolbox, it, within_tolerance), "budget.total")


# ------------------------------------------------------------------ budget.arithmetic


def test_budget_arithmetic(toolbox: Toolbox) -> None:
    it = _itinerary(toolbox, _clean_day(toolbox))
    assert not _fired(_check(toolbox, it, _request()), "budget.arithmetic")
    bad_total = it.model_copy(update={"cost": it.cost.model_copy(
        update={"total": it.cost.total + 1})})
    found = _fired(_check(toolbox, bad_total, _request()), "budget.arithmetic")
    assert found and all(v.severity is Severity.ERROR for v in found)
    bad_day = it.model_copy(update={"days": [it.days[0].model_copy(
        update={"day_cost_aed": Decimal("1")})]})
    msgs = [v.message for v in _fired(_check(toolbox, bad_day, _request()), "budget.arithmetic")]
    assert any("Day 1 cost" in m for m in msgs)


# ------------------------------------------------------------------ time.overlap


def test_time_overlap(toolbox: Toolbox) -> None:
    ok = _itinerary(toolbox, _clean_day(toolbox))
    assert not _fired(_check(toolbox, ok, _request()), "time.overlap")
    clash = [_slot(toolbox, "att_dubai_frame", "09:00", "11:00"),
             _slot(toolbox, "res_xva_cafe", "10:30", "11:30")]
    [v] = _fired(_check(toolbox, _itinerary(toolbox, clash), _request()), "time.overlap")
    assert v.severity is Severity.ERROR and v.slot_index == 1


# ------------------------------------------------------------------ time.opening_hours


def test_opening_hours(toolbox: Toolbox) -> None:
    venue = next(a for a in toolbox.catalog.attractions
                 if a.opening_hours and not a.opening_hours.past_midnight)
    hours = venue.opening_hours
    assert hours is not None
    inside_start = hours.open.replace(minute=0) if hours.open.minute == 0 else hours.open
    inside = _slot(toolbox, venue.id, inside_start.strftime("%H:%M"),
                   time(inside_start.hour + 1, inside_start.minute).strftime("%H:%M"))
    day = next(d for d in (FEB, date(2027, 2, 11), date(2027, 2, 12), date(2027, 2, 13))
               if d.weekday() not in hours.closed_days)
    assert not _fired(_check(toolbox, _itinerary(toolbox, [inside], day=day),
                             _request(day)), "time.opening_hours")
    early = _slot(toolbox, venue.id, "03:00", "04:00")
    [v] = _fired(_check(toolbox, _itinerary(toolbox, [early], day=day), _request(day)),
                 "time.opening_hours")
    assert v.severity is Severity.WARNING


def test_opening_hours_flags_seasonal_closure(toolbox: Toolbox) -> None:
    seasonal = next(a for a in toolbox.catalog.attractions
                    if a.seasonal_months and 7 not in a.seasonal_months)
    it = _itinerary(toolbox, [_slot(toolbox, seasonal.id, "18:00", "19:00")], day=JULY)
    [v] = _fired(_check(toolbox, it, _request(JULY)), "time.opening_hours")
    assert "seasonal" in v.message


# ------------------------------------------------------------------ heat.midday_outdoor


def test_heat_midday_outdoor(toolbox: Toolbox) -> None:
    outdoor = next(a for a in toolbox.catalog.attractions if not a.indoor and not a.seasonal_months)
    midday = [_slot(toolbox, outdoor.id, "12:00", "13:30")]
    [v] = _fired(_check(toolbox, _itinerary(toolbox, midday, day=JULY), _request(JULY)),
                 "heat.midday_outdoor")
    assert v.severity is Severity.ERROR  # July is a hot month → error
    evening = [_slot(toolbox, outdoor.id, "17:00", "18:30")]
    assert not _fired(_check(toolbox, _itinerary(toolbox, evening, day=JULY), _request(JULY)),
                      "heat.midday_outdoor")
    indoor = [_slot(toolbox, "att_dubai_frame", "12:00", "13:00")]
    assert not _fired(_check(toolbox, _itinerary(toolbox, indoor, day=JULY), _request(JULY)),
                      "heat.midday_outdoor")
    assert not _fired(_check(toolbox, _itinerary(toolbox, midday, day=FEB), _request(FEB)),
                      "heat.midday_outdoor")


def test_heat_uses_forecast_and_warns_outside_summer(toolbox: Toolbox) -> None:
    outdoor = next(a for a in toolbox.catalog.attractions if not a.indoor and not a.seasonal_months)
    wx = WeatherContext(source=WeatherSource.FORECAST,
                        days=[DailyWeather(date=OCT, temp_max_c=40.5, temp_min_c=29)],
                        provenance=Provenance(kind=DataKind.API, source="test", as_of=OCT))
    it = _itinerary(toolbox, [_slot(toolbox, outdoor.id, "12:00", "13:00")], day=OCT)
    [v] = _fired(_check(toolbox, it, _request(OCT), ResearchBundle(weather=wx)),
                 "heat.midday_outdoor")
    assert v.severity is Severity.WARNING


# ------------------------------------------------------------------ geo.travel_time


def test_geo_travel_time(toolbox: Toolbox) -> None:
    far = [_slot(toolbox, "att_dubai_frame", "11:00", "12:00", leg_min=120)]
    [v] = _fired(_check(toolbox, _itinerary(toolbox, far), _request()), "geo.travel_time")
    assert v.severity is Severity.WARNING
    near = [_slot(toolbox, "att_dubai_frame", "11:00", "12:00", leg_min=40)]
    assert not _fired(_check(toolbox, _itinerary(toolbox, near), _request()), "geo.travel_time")
    trip = next(a for a in toolbox.catalog.attractions if a.day_trip)
    day_trip = [_slot(toolbox, trip.id, "10:00", "16:00", leg_min=120, indoor=True)]
    assert not _fired(_check(toolbox, _itinerary(toolbox, day_trip), _request()),
                      "geo.travel_time")


# ------------------------------------------------------------------ pace.density


def test_pace_density(toolbox: Toolbox) -> None:
    ids = [a.id for a in toolbox.catalog.attractions if not a.seasonal_months][:4]
    assert len(ids) == 4
    slots = [_slot(toolbox, ref, f"{8 + 2 * i:02d}:00", f"{9 + 2 * i:02d}:00", indoor=True)
             for i, ref in enumerate(ids)]
    relaxed = _request(pace="relaxed")
    [v] = _fired(_check(toolbox, _itinerary(toolbox, slots, request=relaxed), relaxed),
                 "pace.density")
    assert v.severity is Severity.WARNING and v.details["count"] == "4"
    moderate = _request(pace="moderate")
    assert not _fired(_check(toolbox, _itinerary(toolbox, slots, request=moderate), moderate),
                      "pace.density")


# ------------------------------------------------------------------ constraints.dietary


def test_dietary(toolbox: Toolbox) -> None:
    vegan_req = _request(dietary=["vegan"])
    not_vegan = next(r for r in toolbox.catalog.restaurants if "vegan" not in r.dietary)
    vegan = next(r for r in toolbox.catalog.restaurants if "vegan" in r.dietary)
    bad = _itinerary(toolbox, [_slot(toolbox, not_vegan.id, "12:00", "13:00")], request=vegan_req)
    [v] = _fired(_check(toolbox, bad, vegan_req), "constraints.dietary")
    assert v.severity is Severity.ERROR and "vegan" in v.message
    good = _itinerary(toolbox, [_slot(toolbox, vegan.id, "12:00", "13:00")], request=vegan_req)
    assert not _fired(_check(toolbox, good, vegan_req), "constraints.dietary")
    assert not _fired(_check(toolbox, bad, _request()), "constraints.dietary")  # no needs


# ------------------------------------------------------------------ constraints.accessibility


def test_accessibility(toolbox: Toolbox) -> None:
    wheel = _request(accessibility=["wheelchair"])
    no_access = [a for a in toolbox.catalog.attractions if a.wheelchair_accessible is False]
    unknown = [a for a in toolbox.catalog.attractions if a.wheelchair_accessible is None]
    accessible = next(a for a in toolbox.catalog.attractions if a.wheelchair_accessible)
    if no_access:
        it = _itinerary(toolbox, [_slot(toolbox, no_access[0].id, "17:00", "18:00")])
        [v] = _fired(_check(toolbox, it, wheel), "constraints.accessibility")
        assert v.severity is Severity.WARNING
    if unknown:
        it = _itinerary(toolbox, [_slot(toolbox, unknown[0].id, "17:00", "18:00")])
        [v] = _fired(_check(toolbox, it, wheel), "constraints.accessibility")
        assert v.severity is Severity.INFO
    assert no_access or unknown
    ok = _itinerary(toolbox, [_slot(toolbox, accessible.id, "17:00", "18:00")])
    assert not _fired(_check(toolbox, ok, wheel), "constraints.accessibility")
    if no_access:
        it = _itinerary(toolbox, [_slot(toolbox, no_access[0].id, "17:00", "18:00")])
        assert not _fired(_check(toolbox, it, _request()), "constraints.accessibility")


# ------------------------------------------------------------------ culture.ramadan


def test_ramadan(toolbox: Toolbox) -> None:
    ramadan_day = date(2027, 2, 20)  # Ramadan 1448 ≈ 8 Feb - 8 Mar 2027
    it = _itinerary(toolbox, _clean_day(toolbox), day=ramadan_day)
    [v] = _fired(_check(toolbox, it, _request(ramadan_day)), "culture.ramadan")
    assert v.severity is Severity.INFO and "Ramadan" in v.message
    autumn = _itinerary(toolbox, _clean_day(toolbox), day=OCT)
    assert not _fired(_check(toolbox, autumn, _request(OCT)), "culture.ramadan")


# ------------------------------------------------------------------ grounding.known_entities


def test_grounding(toolbox: Toolbox) -> None:
    fake = [_slot(toolbox, "att_imaginary_tower", "10:00", "11:00")]
    [v] = _fired(_check(toolbox, _itinerary(toolbox, fake), _request()),
                 "grounding.known_entities")
    assert v.severity is Severity.WARNING and v.details["ref_id"] == "att_imaginary_tower"
    assert not _fired(_check(toolbox, _itinerary(toolbox, _clean_day(toolbox)), _request()),
                      "grounding.known_entities")


def test_grounding_against_research_bundle(toolbox: Toolbox) -> None:
    """With research supplied, seed venues that were not researched are flagged too."""
    frame = toolbox.catalog.venue("att_dubai_frame")
    research = ResearchBundle(attractions=[frame])  # type: ignore[list-item]
    it = _itinerary(toolbox, _clean_day(toolbox))
    found = _fired(_check(toolbox, it, _request(), research), "grounding.known_entities")
    assert [v.details["ref_id"] for v in found] == ["res_xva_cafe"]


def test_sorted_by_severity_and_has_errors(toolbox: Toolbox) -> None:
    clash = [_slot(toolbox, "att_imaginary_tower", "09:00", "11:00"),
             _slot(toolbox, "res_xva_cafe", "10:30", "11:30")]
    found = _check(toolbox, _itinerary(toolbox, clash), _request())
    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    assert [order[v.severity] for v in found] == sorted(order[v.severity] for v in found)
    assert has_errors(found) and not has_errors([v for v in found if v.severity
                                                 is not Severity.ERROR])


@pytest.mark.parametrize("rule", sorted(RULES))
def test_each_rule_returns_list(toolbox: Toolbox, rule: str) -> None:
    from dubai_planner.validation.rules import RuleContext

    ctx = RuleContext(itinerary=_itinerary(toolbox, _clean_day(toolbox), day=OCT),
                      request=_request(OCT), catalog=toolbox.catalog)
    assert RULES[rule](ctx) == []
