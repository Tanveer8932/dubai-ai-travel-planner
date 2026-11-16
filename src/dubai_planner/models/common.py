"""Shared primitives: money (AED as ``Decimal``), data provenance, geo points, price ranges.

Provenance is first-class: every fact that reaches the user says *where it came from*
and *how much to trust it*. Prices are never presented as confirmed unless their
provenance is :attr:`DataKind.VERIFIED` (which curated seed data never is).
"""

from __future__ import annotations

import math
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, model_validator

CENT = Decimal("0.01")


def to_decimal(value: Any) -> Any:
    """Coerce ints/floats/strings to ``Decimal`` without binary-float artefacts.

    ``Decimal(0.1)`` is ``0.1000000000000000055…``; ``Decimal(str(0.1))`` is ``0.1``.
    Anything that is not number-like is passed through for Pydantic to reject.
    """
    if isinstance(value, bool):  # bool is an int subclass; never money
        return value
    if isinstance(value, float | int):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value.strip())
        except InvalidOperation:
            return value
    return value


def quantize_aed(value: Decimal) -> Decimal:
    """Round to fils (2 dp), half-up, the way prices are quoted in the UAE."""
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


#: A non-negative amount of UAE dirhams, stored as ``Decimal`` with 2 decimal places.
AED = Annotated[
    Decimal,
    BeforeValidator(to_decimal),
    Field(ge=0, allow_inf_nan=False),
    AfterValidator(quantize_aed),
]

#: A possibly negative AED amount (deltas, surpluses/shortfalls).
SignedAED = Annotated[
    Decimal,
    BeforeValidator(to_decimal),
    Field(allow_inf_nan=False),
    AfterValidator(quantize_aed),
]


class StrictModel(BaseModel):
    """Base for all domain models: unknown fields are errors, assignments re-validate."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False)


class DataKind(StrEnum):
    """How a piece of information was obtained, i.e. how far to trust it.

    - ``verified``: checked against an authoritative source (official page, statute,
      central-bank peg) on ``as_of``, with a ``source_url``.
    - ``api``: returned by a live public API (Open-Meteo, OSM, Frankfurter…) on ``as_of``.
    - ``curated``: compiled by hand from public information (e.g. Wikipedia, Visit Dubai,
      OSM) but **not** re-checked live. Facts such as location, category, indoor/outdoor.
    - ``estimated``: an approximation (typical price band, average spend, typical event
      window, approximate Islamic-calendar date). Always shown to users as an estimate.
    """

    VERIFIED = "verified"
    API = "api"
    CURATED = "curated"
    ESTIMATED = "estimated"


class Provenance(StrictModel):
    """Where a fact came from and how reliable it is."""

    kind: DataKind
    source: str = Field(min_length=1, description="Human-readable source name")
    source_url: str | None = None
    as_of: date
    licence: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _verified_needs_url(self) -> Provenance:
        if self.kind is DataKind.VERIFIED and not self.source_url:
            raise ValueError("verified provenance requires a source_url")
        return self

    @property
    def is_estimate(self) -> bool:
        return self.kind is DataKind.ESTIMATED

    def label(self) -> str:
        """Short UI label, e.g. ``estimate · curated seed · as of 2026-11-15``."""
        return f"{self.kind.value} · {self.source} · as of {self.as_of.isoformat()}"


class GeoPoint(StrictModel):
    """WGS-84 coordinate."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)

    def distance_km(self, other: GeoPoint) -> float:
        """Great-circle (haversine) distance in kilometres."""
        r = 6371.0088
        p1, p2 = math.radians(self.lat), math.radians(other.lat)
        dp, dl = p2 - p1, math.radians(other.lon - self.lon)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    def within(self, bbox: tuple[float, float, float, float]) -> bool:
        south, west, north, east = bbox
        return south <= self.lat <= north and west <= self.lon <= east


#: Rough bounding box of the UAE (used to sanity-check seed coordinates).
UAE_BBOX: tuple[float, float, float, float] = (22.5, 51.0, 26.5, 56.5)  # S, W, N, E
#: Greater-Dubai bounding box used for Overpass queries.
DUBAI_BBOX: tuple[float, float, float, float] = (24.79, 54.89, 25.36, 55.65)
#: City centre used for weather queries.
DUBAI_CENTRE = GeoPoint(lat=25.2048, lon=55.2708)


class PriceBand(StrEnum):
    BUDGET = "budget"
    MID = "mid"
    PREMIUM = "premium"
    LUXURY = "luxury"


class PriceUnit(StrEnum):
    PER_PERSON = "per_person"
    PER_ADULT_TICKET = "per_adult_ticket"
    PER_ROOM_NIGHT = "per_room_night"
    PER_TRIP = "per_trip"
    PER_GROUP = "per_group"
    FREE = "free"


class PriceRange(StrictModel):
    """A price as a range with provenance. Seed prices are always ``estimated``."""

    min_aed: AED
    max_aed: AED
    unit: PriceUnit
    provenance: Provenance

    @model_validator(mode="after")
    def _ordered(self) -> PriceRange:
        if self.min_aed > self.max_aed:
            raise ValueError(f"min_aed ({self.min_aed}) > max_aed ({self.max_aed})")
        if self.unit is PriceUnit.FREE and self.max_aed != 0:
            raise ValueError("a FREE price must have max_aed == 0")
        return self

    @property
    def midpoint_aed(self) -> Decimal:
        return quantize_aed((self.min_aed + self.max_aed) / 2)

    @property
    def is_free(self) -> bool:
        return self.max_aed == 0

    @property
    def is_estimate(self) -> bool:
        return self.provenance.kind is not DataKind.VERIFIED

    def label(self) -> str:
        """UI string; anything not verified is prefixed with ``≈`` and called an estimate."""
        if self.is_free:
            return "Free"
        amount = (
            f"AED {self.min_aed:,.0f}"
            if self.min_aed == self.max_aed
            else f"AED {self.min_aed:,.0f}–{self.max_aed:,.0f}"
        )
        if self.is_estimate:
            return f"≈ {amount} (estimate, as of {self.provenance.as_of.isoformat()})"
        return amount
