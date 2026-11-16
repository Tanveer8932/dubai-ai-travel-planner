"""Transport agent (deterministic): modes, a Nol-card recommendation and a transport budget.

It runs in parallel with the other research agents, so it does not know the final hotel
or attractions. It estimates typical legs between representative areas (preferred
areas, else Downtown / Old Dubai / Marina) with the travel tool, then scales by the
pace's legs per day. Airport transfers and Nol cards are added once. Everything is an
estimate (RTA fares from curated seed data, haversine distances). No LLM is used.
"""

from __future__ import annotations

import itertools
from decimal import Decimal
from typing import Any

from pydantic import Field

from dubai_planner.models.common import DataKind, GeoPoint, Provenance, StrictModel, quantize_aed
from dubai_planner.models.request import AccessibilityNeed, TripRequest
from dubai_planner.models.transport import TransportLeg, TransportMode, TransportPlan
from dubai_planner.tools.costs import vehicles_needed
from dubai_planner.tools.search import resolve_areas
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import agent_span
from dubai_planner.validation.thresholds import PACE_LEGS_PER_DAY

AGENT = "transport"
DEFAULT_AREAS = ("downtown", "bur_dubai", "dubai_marina")
AIRPORT_AREA = "al_garhoud"
YOUNG_CHILD_AGE = 5


class TransportInput(StrictModel):
    request: TripRequest
    hotel_area: str | None = None
    areas: list[str] = Field(default_factory=list, description="Area ids likely to be visited")


def prefers_taxi(request: TripRequest) -> bool:
    """Door-to-door taxis for wheelchair/limited-mobility needs or very young children."""
    needs = {AccessibilityNeed.WHEELCHAIR, AccessibilityNeed.LIMITED_MOBILITY}
    young = any(age < YOUNG_CHILD_AGE for age in request.travellers.children_ages)
    return bool(needs & set(request.accessibility)) or young


def _points(data: TransportInput, tools: Toolbox) -> list[tuple[str, GeoPoint]]:
    wanted = list(data.areas) or sorted(resolve_areas(data.request, tools.catalog))
    wanted = [a for a in wanted if (area := tools.catalog.area(a)) and not area.day_trip]
    if data.hotel_area:
        wanted.insert(0, data.hotel_area)
    for default in DEFAULT_AREAS:
        if len(set(wanted)) >= 3:
            break
        wanted.append(default)
    points: list[tuple[str, GeoPoint]] = []
    for area_id in dict.fromkeys(wanted):
        area = tools.catalog.area(area_id)
        if area is not None:
            points.append((area_id, area.centroid))
    return points


def run(data: TransportInput, llm: Any = None, tools: Toolbox | None = None) -> TransportPlan:
    """``llm`` is accepted for a uniform agent signature but not used."""
    tb = tools or Toolbox.default()
    request = data.request
    party = request.travellers
    with agent_span(AGENT) as span:
        taxi_first = prefers_taxi(request)
        legs: list[TransportLeg] = [
            tb.estimate_travel(a, b, travellers=party, prefer_taxi=taxi_first,
                               from_ref=a_id, to_ref=b_id)
            for (a_id, a), (b_id, b) in itertools.combinations(_points(data, tb), 2)
        ]
        avg_leg = quantize_aed(sum((leg.cost_aed for leg in legs), Decimal("0")) / len(legs))
        daily = quantize_aed(avg_leg * PACE_LEGS_PER_DAY[request.pace])

        seed = tb.catalog.transport
        extras = Decimal("0")
        notes: list[str] = []
        airport = tb.catalog.area(AIRPORT_AREA)
        hotel = tb.catalog.area(data.hotel_area or DEFAULT_AREAS[0])
        if airport is not None and hotel is not None and request.nights > 0:
            km = airport.centroid.distance_km(hotel.centroid) * seed.speeds.road_detour_factor
            transfer = seed.taxi.estimate(km, from_airport=True) * vehicles_needed(party)
            extras += 2 * transfer
            notes.append(f"Airport taxi transfers (both ways) ≈ AED {2 * transfer:,.0f} "
                         "(estimate).")
        modes = list(dict.fromkeys(leg.mode for leg in legs)) or [TransportMode.TAXI]
        nol = TransportMode.METRO in modes
        if nol and seed.nol_fares.nol_card_cost:
            cards = seed.nol_fares.nol_card_cost * party.total
            extras += cards
            notes.append(f"Nol cards for {party.total} traveller(s) ≈ AED {cards:,.0f} "
                         "(card cost; estimate).")
        if taxi_first:
            notes.append("Taxis preferred for door-to-door travel (accessibility or young "
                         "children).")
        notes.append("Taxi estimates exclude Salik tolls and peak/night surcharges.")
        total = quantize_aed(daily * request.num_days + extras)
        plan = TransportPlan(
            primary_modes=modes,
            nol_card_recommended=nol,
            estimated_daily_cost_aed=daily,
            estimated_total_cost_aed=total,
            legs=legs,
            notes=notes,
            provenance=Provenance(
                kind=DataKind.ESTIMATED,
                source="Planner estimate: curated RTA fares and taxi tariff, haversine distance",
                as_of=seed.nol_fares.provenance.as_of,
                note="Not live fares; check rta.ae before travelling.",
            ),
        )
        span.update(modes=[m.value for m in modes], daily=str(daily), total=str(total))
        return plan
