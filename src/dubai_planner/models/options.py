"""Venue models: attractions, hotels, restaurants and areas.

The same models validate curated seed data (``data/seed/*.json``) and the research
agents' outputs, so agents can only return venues that exist in tool results
(they copy records by ``id``; ``score``/``rationale`` are the only fields they add).
"""

from __future__ import annotations

from datetime import time
from enum import IntEnum, StrEnum

from pydantic import Field, model_validator

from dubai_planner.models.common import (
    GeoPoint,
    PriceBand,
    PriceRange,
    Provenance,
    StrictModel,
)
from dubai_planner.models.request import DietaryFlag

#: IDs are lowercase slugs with a type prefix, e.g. ``att_burj_khalifa``.
ID_PATTERN = r"^[a-z]{2,5}_[a-z0-9_]+$"


class Weekday(IntEnum):
    MON = 0
    TUE = 1
    WED = 2
    THU = 3
    FRI = 4
    SAT = 5
    SUN = 6


class TimeOfDay(StrEnum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


class OpeningHours(StrictModel):
    """Simplified daily opening hours.

    ``close`` earlier than ``open`` means the venue closes after midnight. ``osm`` keeps
    the raw OpenStreetMap ``opening_hours`` string when the record was enriched from OSM.
    """

    open: time
    close: time
    closed_days: list[Weekday] = Field(default_factory=list)
    osm: str | None = None

    @property
    def past_midnight(self) -> bool:
        return self.close <= self.open

    def is_open(self, weekday: Weekday, start: time, end: time) -> bool:
        """True when ``[start, end]`` on ``weekday`` lies within the opening hours."""
        if weekday in self.closed_days:
            return False
        if self.past_midnight:
            return start >= self.open or end <= self.close
        return self.open <= start and end <= self.close


class AttractionCategory(StrEnum):
    LANDMARK = "landmark"
    MUSEUM = "museum"
    CULTURE = "culture"
    SHOPPING = "shopping"
    MARKET = "market"
    BEACH = "beach"
    DESERT = "desert"
    THEME_PARK = "theme_park"
    WATER_PARK = "water_park"
    NATURE = "nature"
    PARK = "park"
    RELIGIOUS = "religious"
    ENTERTAINMENT = "entertainment"
    ADVENTURE = "adventure"
    CRUISE = "cruise"
    ART = "art"


class Venue(StrictModel):
    """Fields shared by every venue record."""

    id: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=2)
    area: str = Field(description="Area id from areas.json")
    location: GeoPoint
    description: str = ""
    family_friendly: bool = True
    wheelchair_accessible: bool | None = Field(
        default=None, description="None = unknown (not verified)"
    )
    opening_hours: OpeningHours | None = None
    tags: list[str] = Field(default_factory=list)
    osm_id: str | None = Field(default=None, description="e.g. 'way/386649647'")
    provenance: Provenance
    enrichment: Provenance | None = Field(
        default=None, description="Set when location/hours were refreshed from OSM"
    )
    # Filled in by research agents; never present in seed data.
    score: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = None


class AttractionOption(Venue):
    category: AttractionCategory
    indoor: bool
    typical_duration_min: int = Field(ge=10, le=720)
    price: PriceRange
    best_time: list[TimeOfDay] = Field(default_factory=list)
    seasonal_months: list[int] = Field(
        default_factory=list, description="Months open (1-12); empty = all year"
    )
    requires_booking: bool = False
    day_trip: bool = False

    @model_validator(mode="after")
    def _months_valid(self) -> AttractionOption:
        if any(m < 1 or m > 12 for m in self.seasonal_months):
            raise ValueError("seasonal_months must be 1..12")
        return self

    def open_in_month(self, month: int) -> bool:
        return not self.seasonal_months or month in self.seasonal_months


class HotelOption(Venue):
    stars: int = Field(ge=1, le=5)
    price_band: PriceBand
    nightly_rate: PriceRange
    nearest_metro: str | None = None
    amenities: list[str] = Field(default_factory=list)


class MealType(StrEnum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class DiningOption(Venue):
    cuisines: list[str] = Field(min_length=1)
    dietary: list[DietaryFlag] = Field(
        default_factory=list, description="Dietary needs the menu can cater for"
    )
    fully_vegetarian: bool = False
    price_band: PriceBand
    avg_spend: PriceRange
    meal_types: list[MealType] = Field(min_length=1)

    @model_validator(mode="after")
    def _veg_consistency(self) -> DiningOption:
        if self.fully_vegetarian and DietaryFlag.VEGETARIAN not in self.dietary:
            raise ValueError("fully_vegetarian venues must list the 'vegetarian' flag")
        return self

    def caters_for(self, needs: list[DietaryFlag]) -> bool:
        return all(flag in self.dietary for flag in needs)


class Area(StrictModel):
    """A neighbourhood or day-trip destination, used to cluster days geographically."""

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    centroid: GeoPoint
    description: str
    nearest_metro: list[str] = Field(default_factory=list)
    day_trip: bool = False
    tags: list[str] = Field(default_factory=list)
    provenance: Provenance
