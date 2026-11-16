"""Travel-time and fare estimation (deterministic; seed fares and speeds, haversine distance).

Road distance = straight-line distance × ``road_detour_factor``. Mode choice when not
forced: walk ≤ 1 km; metro when both ends are within reach of a station and the trip
is worth it; otherwise taxi. All fares are estimates from the curated RTA tables.
"""

from __future__ import annotations

from decimal import Decimal

from dubai_planner.models.common import GeoPoint, quantize_aed
from dubai_planner.models.request import Travellers
from dubai_planner.models.transport import MetroStation, TransportLeg, TransportMode, TransportSeed
from dubai_planner.tools.costs import vehicles_needed

WALK_MAX_KM = 1.0
STATION_REACH_KM = 1.5
METRO_MIN_KM = 3.0
HIGHWAY_FROM_KM = 25.0
TAXI_OVERHEAD_MIN = 5
MAX_DISTANCE_KM = 500.0
MAX_DURATION_MIN = 600


def nearest_station(point: GeoPoint, stations: list[MetroStation]) -> tuple[MetroStation, float]:
    if not stations:
        raise ValueError("no metro stations in seed data")
    best = min(stations, key=lambda s: point.distance_km(s.location))
    return best, point.distance_km(best.location)


def metro_zones(road_km: float) -> int:
    """Approximate Nol zones crossed from trip length (RTA zones are ~10-15 km wide)."""
    if road_km < 10:
        return 1
    return 2 if road_km < 25 else 3


def metro_fare(transport: TransportSeed, zones: int, *, gold: bool = False) -> Decimal:
    """Single-journey Nol fare per person (Silver or Gold class)."""
    return transport.nol_fares.fare(zones, gold=gold)


def _minutes(km: float, kmh: float) -> float:
    return km / kmh * 60


def estimate_travel(
    origin: GeoPoint,
    destination: GeoPoint,
    transport: TransportSeed,
    *,
    travellers: Travellers | None = None,
    mode: TransportMode | None = None,
    prefer_taxi: bool = False,
    from_ref: str | None = None,
    to_ref: str | None = None,
) -> TransportLeg:
    """Estimated leg between two points for the whole party (fares × people/vehicles)."""
    party = travellers or Travellers()
    speeds = transport.speeds
    straight = origin.distance_km(destination)
    road = min(straight * speeds.road_detour_factor, MAX_DISTANCE_KM)
    stations = transport.metro_stations
    if mode is None:
        if straight <= WALK_MAX_KM:
            mode = TransportMode.WALK
        elif prefer_taxi or road < METRO_MIN_KM or not stations:
            mode = TransportMode.TAXI
        else:
            _, d_from = nearest_station(origin, stations)
            _, d_to = nearest_station(destination, stations)
            reachable = d_from <= STATION_REACH_KM and d_to <= STATION_REACH_KM
            mode = TransportMode.METRO if reachable else TransportMode.TAXI

    if mode is TransportMode.WALK:
        distance = straight * speeds.road_detour_factor
        minutes = _minutes(distance, speeds.walk_kmh)
        cost = Decimal("0")
    elif mode is TransportMode.METRO:
        _, d_from = nearest_station(origin, stations)
        _, d_to = nearest_station(destination, stations)
        distance = road
        minutes = (speeds.metro_access_min + _minutes(road, speeds.metro_kmh)
                   + _minutes(d_from + d_to, speeds.walk_kmh))
        cost = metro_fare(transport, metro_zones(road)) * party.total
    else:  # taxi / ride-hail / anything road-based
        distance = road
        kmh = speeds.taxi_highway_kmh if road >= HIGHWAY_FROM_KM else speeds.taxi_urban_kmh
        minutes = TAXI_OVERHEAD_MIN + _minutes(road, kmh)
        cost = transport.taxi.estimate(road) * vehicles_needed(party)
        if mode is TransportMode.RIDE_HAIL:
            cost = cost * transport.taxi.ride_hail_multiplier_max
        if mode is TransportMode.TOUR_PICKUP:
            cost = Decimal("0")  # included in the tour price
    return TransportLeg(
        mode=mode,
        from_ref=from_ref,
        to_ref=to_ref,
        distance_km=round(min(distance, MAX_DISTANCE_KM), 2),
        duration_min=min(round(minutes), MAX_DURATION_MIN),
        cost_aed=quantize_aed(cost),
        cost_is_estimate=True,
    )
