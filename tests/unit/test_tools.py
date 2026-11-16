"""Deterministic tools: seed search (hard filters, scoring), travel, fares, currency, costs."""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal

import pytest
from factories import price, trip_request

from dubai_planner.models import (
    Activity,
    ActivityKind,
    BudgetCategory,
    DayPlan,
    DietaryFlag,
    GeoPoint,
    PriceUnit,
    TimeSlot,
    TransportMode,
    Travellers,
)
from dubai_planner.tools import costs, search, travel
from dubai_planner.tools.toolbox import TOOL_SPECS, Toolbox

# ------------------------------------------------------------------ search


def test_attraction_search_is_deterministic_and_scored(toolbox: Toolbox) -> None:
    req = trip_request()
    first = toolbox.search_attractions(req, limit=20)
    second = toolbox.search_attractions(req, limit=20)
    assert [a.id for a in first] == [a.id for a in second]
    scores = [a.score for a in first]
    assert all(s is not None and 0 <= s <= 1 for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert all(toolbox.catalog.venue(a.id) is not None for a in first)  # grounded copies


def test_interests_change_ranking(toolbox: Toolbox) -> None:
    history = toolbox.search_attractions(trip_request(interests=["history"]), limit=5)
    beach = toolbox.search_attractions(trip_request(interests=["beach"]), limit=5)
    assert [a.id for a in history] != [a.id for a in beach]
    assert any(a.category.value == "beach" or "beach" in a.tags for a in beach)


def test_must_include_ranks_first(toolbox: Toolbox) -> None:
    found = toolbox.search_attractions(trip_request(must_include=["Dubai Frame"]), limit=5)
    assert found[0].id == "att_dubai_frame" and found[0].score == 1.0


def test_avoid_terms_exclude(toolbox: Toolbox) -> None:
    found = toolbox.search_attractions(trip_request(avoid=["desert"], travellers=Travellers()),
                                       limit=60)
    assert not any(a.category.value == "desert" for a in found)


def test_children_exclude_non_family_venues(toolbox: Toolbox) -> None:
    found = toolbox.search_attractions(trip_request(), limit=60)  # factory has a child
    assert all(a.family_friendly for a in found)


def test_wheelchair_excludes_known_inaccessible(toolbox: Toolbox) -> None:
    req = trip_request(accessibility=["wheelchair"])
    for found in (toolbox.search_attractions(req, limit=60), toolbox.search_hotels(req),
                  toolbox.search_restaurants(req, limit=60)):
        assert all(v.wheelchair_accessible is not False for v in found)


def test_seasonal_attractions_excluded_out_of_season(toolbox: Toolbox) -> None:
    seasonal = [a for a in toolbox.catalog.attractions if a.seasonal_months]
    assert seasonal, "seed should contain seasonal attractions"
    july = trip_request(start_date=date(2027, 7, 10), end_date=date(2027, 7, 12))
    found = {a.id for a in toolbox.search_attractions(july, limit=60)}
    assert all(a.id not in found for a in seasonal if 7 not in a.seasonal_months)


def test_restaurants_meet_dietary_needs(toolbox: Toolbox) -> None:
    req = trip_request(dietary=["vegan", "gluten_free"], travellers=Travellers())
    found = toolbox.search_restaurants(req, limit=60)
    assert found and all(r.caters_for([DietaryFlag.VEGAN, DietaryFlag.GLUTEN_FREE])
                         for r in found)


def test_hotels_track_budget(toolbox: Toolbox) -> None:
    cheap = toolbox.search_hotels(trip_request(budget_aed=Decimal("2500")))[0]
    rich = toolbox.search_hotels(trip_request(budget_aed=Decimal("60000")))[0]
    assert cheap.nightly_rate.midpoint_aed < rich.nightly_rate.midpoint_aed
    assert not any(toolbox.catalog.area(h.area).day_trip  # type: ignore[union-attr]
                   for h in toolbox.search_hotels(trip_request()))


def test_no_hotels_for_day_visit(toolbox: Toolbox) -> None:
    day = trip_request(start_date=date(2027, 2, 10), end_date=date(2027, 2, 10))
    assert toolbox.search_hotels(day) == []


def test_preferred_area_resolution(toolbox: Toolbox) -> None:
    req = trip_request(preferred_areas=["Marina", "old dubai"])
    assert "dubai_marina" in search.resolve_areas(req, toolbox.catalog)


# ------------------------------------------------------------------ travel / fares


def test_walk_metro_taxi_modes(toolbox: Toolbox) -> None:
    seed = toolbox.catalog.transport
    stations = seed.metro_stations
    a, b = stations[0].location, stations[-1].location
    near = GeoPoint(lat=a.lat + 0.002, lon=a.lon)
    assert travel.estimate_travel(a, near, seed).mode is TransportMode.WALK
    long_leg = travel.estimate_travel(a, b, seed, travellers=Travellers(adults=2))
    assert long_leg.mode is TransportMode.METRO
    assert long_leg.cost_aed == travel.metro_fare(seed, travel.metro_zones(
        long_leg.distance_km)) * 2
    taxi = travel.estimate_travel(a, b, seed, prefer_taxi=True)
    assert taxi.mode is TransportMode.TAXI and taxi.cost_aed >= seed.taxi.minimum_fare_aed
    assert all(leg.cost_is_estimate for leg in (long_leg, taxi))


def test_taxi_cost_scales_with_vehicles(toolbox: Toolbox) -> None:
    seed = toolbox.catalog.transport
    a, b = GeoPoint(lat=25.2, lon=55.27), GeoPoint(lat=25.08, lon=55.14)
    one = travel.estimate_travel(a, b, seed, mode=TransportMode.TAXI)
    six = travel.estimate_travel(a, b, seed, mode=TransportMode.TAXI,
                                 travellers=Travellers(adults=6))
    assert six.cost_aed == one.cost_aed * 2


def test_tour_pickup_is_free_and_ride_hail_dearer(toolbox: Toolbox) -> None:
    seed = toolbox.catalog.transport
    a, b = GeoPoint(lat=25.2, lon=55.27), GeoPoint(lat=25.0, lon=55.5)
    assert travel.estimate_travel(a, b, seed, mode=TransportMode.TOUR_PICKUP).cost_aed == 0
    taxi = travel.estimate_travel(a, b, seed, mode=TransportMode.TAXI).cost_aed
    assert travel.estimate_travel(a, b, seed, mode=TransportMode.RIDE_HAIL).cost_aed > taxi


@pytest.mark.parametrize(("km", "zones"), [(3, 1), (9.9, 1), (12, 2), (30, 3)])
def test_metro_zones(km: float, zones: int) -> None:
    assert travel.metro_zones(km) == zones


def test_metro_fare_tool(toolbox: Toolbox) -> None:
    fares = toolbox.catalog.transport.nol_fares
    assert toolbox.metro_fare(1) == fares.silver_by_zones[0]
    assert toolbox.metro_fare(3, gold=True) == fares.gold_by_zones[2]


def test_estimate_between_ids(toolbox: Toolbox) -> None:
    leg = toolbox.estimate_travel_between("htl_rove_downtown", "att_dubai_frame")
    assert leg.from_ref == "htl_rove_downtown" and leg.to_ref == "att_dubai_frame"
    with pytest.raises(ValueError, match="unknown venue"):
        toolbox.estimate_travel_between("att_nowhere", "att_dubai_frame")


def test_nearest_station_requires_stations() -> None:
    with pytest.raises(ValueError, match="no metro stations"):
        travel.nearest_station(GeoPoint(lat=25.2, lon=55.3), [])


def test_convert_currency_tool_uses_peg_offline(toolbox: Toolbox) -> None:
    assert toolbox.convert_currency(100, "USD") == Decimal("367.25")
    assert toolbox.convert_currency("367.25", "AED", "USD") == Decimal("100.00")


# ------------------------------------------------------------------ costs


def test_party_cost_units() -> None:
    party = Travellers(adults=2, children_ages=[5])
    assert costs.party_cost(price("10", "20"), party) == Decimal("45.00")
    assert costs.party_cost(price("100", "100", PriceUnit.PER_GROUP), party) == Decimal("100.00")
    assert costs.party_cost(price("0", "0", PriceUnit.FREE), party) == 0
    assert costs.party_cost(price("200", "300", PriceUnit.PER_ROOM_NIGHT),
                            Travellers(adults=3)) == Decimal("500.00")


def test_rooms_and_hotel_cost(toolbox: Toolbox) -> None:
    hotel = toolbox.search_hotels(trip_request())[0]
    assert costs.rooms_needed(Travellers(adults=3)) == 2
    assert costs.hotel_cost(hotel, 0, Travellers()) == 0
    assert costs.hotel_cost(hotel, 3, Travellers(adults=4)) == (
        hotel.nightly_rate.midpoint_aed * 3 * 2)


def test_food_rates_increase_with_band(toolbox: Toolbox) -> None:
    rates = list(costs.food_daily_rates(toolbox.catalog).values())
    assert rates == sorted(rates) and rates[0] > 0


def _day(*slots: TimeSlot) -> DayPlan:
    return DayPlan(date=date(2027, 2, 10), slots=list(slots))


def _slot(start: int, kind: ActivityKind, cost: str, leg_cost: str = "0") -> TimeSlot:
    from dubai_planner.models import TransportLeg

    return TimeSlot(start=time(start), end=time(start + 1),
                    activity=Activity(kind=kind, name=f"x{start}", ref_id=f"att_x{start}"
                                      if kind is not ActivityKind.FREE_TIME else None),
                    transport_from_prev=TransportLeg(mode=TransportMode.TAXI, distance_km=3,
                                                     duration_min=10,
                                                     cost_aed=Decimal(leg_cost)),
                    cost_aed=Decimal(cost), indoor=True)


def test_calculate_costs_is_exact_and_consistent(toolbox: Toolbox) -> None:
    hotel = toolbox.search_hotels(trip_request())[0]
    day = _day(_slot(9, ActivityKind.ATTRACTION, "100.10", "12.45"),
               _slot(13, ActivityKind.DINING, "80.20", "0.35"))
    breakdown = costs.calculate_costs([day], hotel=hotel, nights=2, travellers=Travellers(),
                                      budget_aed=Decimal("5000"), contingency_pct=Decimal("0.1"))
    assert breakdown.activities == Decimal("100.10")
    assert breakdown.food == Decimal("80.20")
    assert breakdown.transport == Decimal("12.80")
    subtotal = breakdown.accommodation + Decimal("193.10")
    assert breakdown.contingency == (subtotal * Decimal("0.1")).quantize(Decimal("0.01"))
    assert breakdown.is_consistent and breakdown.total == breakdown.category_sum
    assert costs.day_cost(day) == Decimal("193.10")
    assert all(i.kind.value == "estimated" for i in breakdown.items)
    assert {i.category for i in breakdown.items} == set(BudgetCategory)


# ------------------------------------------------------------------ registry


def test_registry_lists_all_plan_tools() -> None:
    assert set(TOOL_SPECS) == {
        "search_attractions", "search_hotels", "search_restaurants", "get_weather_context",
        "get_events", "estimate_travel", "metro_fare", "convert_currency", "calculate_costs"}
    for spec in TOOL_SPECS.values():
        assert spec.description and spec.args_model.model_json_schema()["properties"]


def test_registry_call_validates_arguments(toolbox: Toolbox) -> None:
    assert toolbox.call("metro_fare", {"zones": 2}) == toolbox.metro_fare(2)
    assert toolbox.call("convert_currency", {"amount": "10", "from_currency": "USD"}) == Decimal(
        "36.73")
    leg = toolbox.call("estimate_travel", {"from_ref": "downtown", "to_ref": "dubai_marina",
                                           "adults": 2, "children": 1})
    assert leg.distance_km > 10
    with pytest.raises(KeyError, match="unknown tool"):
        toolbox.call("book_flight", {})
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        toolbox.call("metro_fare", {"zones": 9})


def test_registry_weather_events_and_search(toolbox: Toolbox) -> None:
    req = trip_request()
    wx = toolbox.call("get_weather_context", {"start": req.start_date, "end": req.end_date})
    assert len(wx.days) == req.num_days
    events = toolbox.call("get_events", {"start": "2027-02-01", "end": "2027-03-15"})
    assert any(e.id.startswith("cal_ramadan") for e in events)
    found = toolbox.call("search_restaurants", {"request": req.model_dump(), "limit": 3})
    assert len(found) == 3


def test_registry_calculate_costs(toolbox: Toolbox) -> None:
    req = trip_request()
    day = _day(_slot(9, ActivityKind.ATTRACTION, "50"))
    out = toolbox.call("calculate_costs", {"request": req.model_dump(), "days": [day.model_dump()],
                                           "hotel_id": "htl_rove_downtown"})
    assert out.accommodation > 0 and out.is_consistent
    with pytest.raises(ValueError, match="unknown hotel"):
        toolbox.call("calculate_costs", {"request": req.model_dump(), "days": [],
                                         "hotel_id": "att_dubai_frame"})


def test_osm_and_wikivoyage_tools_fall_back_offline(toolbox: Toolbox) -> None:
    from dubai_planner.data.overpass import PoiCategory

    pois = toolbox.find_osm_pois(PoiCategory.METRO_STATION)
    assert pois and all(p.ref_id for p in pois)  # seed-derived fallback, grounded
    guide = toolbox.get_area_guide("Dubai/Deira")  # never raises offline
    assert guide is None or guide.extract
