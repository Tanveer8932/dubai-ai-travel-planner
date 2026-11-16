"""User trip request: the structured form of a natural-language planning request."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from pydantic import Field, model_validator

from dubai_planner.models.common import AED, StrictModel, quantize_aed

MAX_TRIP_DAYS = 30


class BudgetScope(StrEnum):
    """How ``budget_aed`` should be interpreted. Flights are always out of scope."""

    TOTAL = "total"  # whole trip, all travellers
    PER_PERSON = "per_person"  # whole trip, per traveller
    PER_DAY = "per_day"  # per day, all travellers


class Pace(StrEnum):
    RELAXED = "relaxed"
    MODERATE = "moderate"
    PACKED = "packed"


class DietaryFlag(StrEnum):
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    HALAL = "halal"
    GLUTEN_FREE = "gluten_free"


class AccessibilityNeed(StrEnum):
    WHEELCHAIR = "wheelchair"
    LIMITED_MOBILITY = "limited_mobility"
    STROLLER = "stroller"


class Travellers(StrictModel):
    adults: int = Field(default=1, ge=1, le=20)
    children_ages: list[int] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def _child_ages_valid(self) -> Travellers:
        if any(age < 0 or age > 17 for age in self.children_ages):
            raise ValueError("children_ages must be between 0 and 17")
        return self

    @property
    def children(self) -> int:
        return len(self.children_ages)

    @property
    def total(self) -> int:
        return self.adults + self.children


class TripRequest(StrictModel):
    """A complete, validated trip request (output of the Requirements agent)."""

    start_date: date
    end_date: date
    travellers: Travellers = Field(default_factory=Travellers)
    budget_aed: AED = Field(gt=0)
    budget_scope: BudgetScope = BudgetScope.TOTAL
    interests: list[str] = Field(default_factory=list)
    pace: Pace = Pace.MODERATE
    dietary: list[DietaryFlag] = Field(default_factory=list)
    accessibility: list[AccessibilityNeed] = Field(default_factory=list)
    preferred_areas: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _date_range(self) -> TripRequest:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        if self.num_days > MAX_TRIP_DAYS:
            raise ValueError(f"trips longer than {MAX_TRIP_DAYS} days are not supported")
        return self

    @property
    def num_days(self) -> int:
        """Inclusive day count (a 3-night stay starting Mon and ending Thu is 4 days)."""
        return (self.end_date - self.start_date).days + 1

    @property
    def nights(self) -> int:
        return (self.end_date - self.start_date).days

    def total_budget_aed(self) -> Decimal:
        """Budget for the whole trip and all travellers, whatever the scope."""
        if self.budget_scope is BudgetScope.PER_PERSON:
            return quantize_aed(self.budget_aed * self.travellers.total)
        if self.budget_scope is BudgetScope.PER_DAY:
            return quantize_aed(self.budget_aed * self.num_days)
        return self.budget_aed


class TripRequestDraft(StrictModel):
    """Partial request used before clarification; all fields optional.

    The clarification gate asks the user for :meth:`missing_fields`, then
    :meth:`to_request` produces the strict :class:`TripRequest`.
    """

    start_date: date | None = None
    end_date: date | None = None
    travellers: Travellers | None = None
    budget_aed: AED | None = None
    budget_scope: BudgetScope | None = None
    interests: list[str] = Field(default_factory=list)
    pace: Pace | None = None
    dietary: list[DietaryFlag] = Field(default_factory=list)
    accessibility: list[AccessibilityNeed] = Field(default_factory=list)
    preferred_areas: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    notes: str = ""

    REQUIRED: ClassVar[tuple[str, ...]] = ("start_date", "end_date", "budget_aed")

    def missing_fields(self) -> list[str]:
        return [name for name in self.REQUIRED if getattr(self, name) is None]

    @property
    def is_complete(self) -> bool:
        return not self.missing_fields()

    def to_request(self) -> TripRequest:
        """Build the strict request; raises ``ValueError`` listing missing fields."""
        missing = self.missing_fields()
        if missing:
            raise ValueError(f"trip request incomplete, missing: {', '.join(missing)}")
        data = self.model_dump(exclude_none=True)
        return TripRequest.model_validate(data)
