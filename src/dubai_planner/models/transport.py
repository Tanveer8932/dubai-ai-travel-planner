"""Transport models: legs between activities, a trip transport plan, and fare/speed seed data."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import Field, model_validator

from dubai_planner.models.common import AED, GeoPoint, Provenance, StrictModel, quantize_aed


class TransportMode(StrEnum):
    WALK = "walk"
    METRO = "metro"
    TRAM = "tram"
    BUS = "bus"
    TAXI = "taxi"
    RIDE_HAIL = "ride_hail"
    ABRA = "abra"
    MONORAIL = "monorail"
    TOUR_PICKUP = "tour_pickup"  # transfer included in a tour (desert safari, day trip)
    INTERCITY_BUS = "intercity_bus"


class TransportLeg(StrictModel):
    """Movement from the previous slot's location to this slot's location."""

    mode: TransportMode
    from_ref: str | None = None
    to_ref: str | None = None
    distance_km: float = Field(ge=0, le=500)
    duration_min: int = Field(ge=0, le=600)
    cost_aed: AED = Decimal("0")
    cost_is_estimate: bool = True


class TransportPlan(StrictModel):
    """Output of the Transport agent: modes to use and an estimated transport budget."""

    primary_modes: list[TransportMode] = Field(min_length=1)
    nol_card_recommended: bool = True
    estimated_daily_cost_aed: AED
    estimated_total_cost_aed: AED
    legs: list[TransportLeg] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    provenance: Provenance


# --------------------------------------------------------------------------- seed data


class MetroLine(StrEnum):
    RED = "red"
    GREEN = "green"
    TRAM = "tram"


class MetroStation(StrictModel):
    id: str = Field(pattern=r"^stn_[a-z0-9_]+$")
    name: str
    lines: list[MetroLine] = Field(min_length=1)
    location: GeoPoint
    area: str | None = None
    provenance: Provenance


class NolFareTable(StrictModel):
    """Metro/tram/bus fares by number of zones crossed (index 0 = within 1 zone)."""

    silver_by_zones: list[AED] = Field(min_length=3, max_length=3)
    gold_by_zones: list[AED] = Field(min_length=3, max_length=3)
    day_pass_silver: AED | None = None
    nol_card_cost: AED | None = None
    provenance: Provenance

    def fare(self, zones: int, gold: bool = False) -> Decimal:
        table = self.gold_by_zones if gold else self.silver_by_zones
        return table[min(max(zones, 1), 3) - 1]


class TaxiTariff(StrictModel):
    flag_fall_aed: AED
    per_km_aed: AED
    minimum_fare_aed: AED
    airport_flag_fall_aed: AED
    ride_hail_multiplier_min: Decimal = Field(ge=1)
    ride_hail_multiplier_max: Decimal = Field(ge=1)
    provenance: Provenance

    def estimate(self, distance_km: float, from_airport: bool = False) -> Decimal:
        flag = self.airport_flag_fall_aed if from_airport else self.flag_fall_aed
        raw = flag + self.per_km_aed * Decimal(str(round(distance_km, 2)))
        return quantize_aed(max(raw, self.minimum_fare_aed))


class FixedFare(StrictModel):
    id: str
    mode: TransportMode
    description: str
    fare: AED
    provenance: Provenance


class TravelSpeeds(StrictModel):
    """Average door-to-door speeds (km/h) used by the haversine travel-time estimator."""

    walk_kmh: float = Field(gt=0)
    metro_kmh: float = Field(gt=0)
    taxi_urban_kmh: float = Field(gt=0)
    taxi_highway_kmh: float = Field(gt=0)
    road_detour_factor: float = Field(ge=1, description="road distance / straight line")
    metro_access_min: int = Field(ge=0, description="walk + wait overhead per metro leg")
    provenance: Provenance


class TransportSeed(StrictModel):
    nol_fares: NolFareTable
    taxi: TaxiTariff
    fixed_fares: list[FixedFare]
    speeds: TravelSpeeds
    metro_stations: list[MetroStation]

    @model_validator(mode="after")
    def _unique_station_ids(self) -> TransportSeed:
        ids = [s.id for s in self.metro_stations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate metro station ids")
        return self
