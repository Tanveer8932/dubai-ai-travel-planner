"""The shared itinerary contract produced by both orchestration modes."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from dubai_planner.models.budget import CostBreakdown
from dubai_planner.models.common import AED, GeoPoint, StrictModel, quantize_aed
from dubai_planner.models.options import HotelOption
from dubai_planner.models.transport import TransportLeg


class PlannerMode(StrEnum):
    LANGGRAPH = "langgraph"
    CREWAI_SEQUENTIAL = "crewai_sequential"
    CREWAI_HIERARCHICAL = "crewai_hierarchical"


class ActivityKind(StrEnum):
    ATTRACTION = "attraction"
    DINING = "dining"
    HOTEL = "hotel"  # check-in / rest at hotel
    TRANSIT = "transit"
    FREE_TIME = "free_time"
    EVENT = "event"


#: Kinds that refer to a concrete venue and therefore must carry a seed ``ref_id``.
GROUNDED_KINDS = frozenset({ActivityKind.ATTRACTION, ActivityKind.DINING, ActivityKind.HOTEL})


class Activity(StrictModel):
    kind: ActivityKind
    name: str = Field(min_length=1)
    ref_id: str | None = Field(default=None, description="Seed/tool-result id of the venue")
    area: str | None = None
    location: GeoPoint | None = None
    description: str = ""

    @model_validator(mode="after")
    def _grounded(self) -> Activity:
        if self.kind in GROUNDED_KINDS and not self.ref_id:
            raise ValueError(f"{self.kind.value} activities must reference a venue ref_id")
        return self


class TimeSlot(StrictModel):
    start: time
    end: time
    activity: Activity
    transport_from_prev: TransportLeg | None = None
    cost_aed: AED = Decimal("0")
    indoor: bool

    @model_validator(mode="after")
    def _start_before_end(self) -> TimeSlot:
        if self.end <= self.start:
            raise ValueError(
                f"slot end {self.end} must be after start {self.start} (no midnight spans)"
            )
        return self

    @property
    def duration_min(self) -> int:
        delta = datetime.combine(date.min, self.end) - datetime.combine(date.min, self.start)
        return int(delta.total_seconds() // 60)


class DayPlan(StrictModel):
    date: date
    theme: str = ""
    slots: list[TimeSlot] = Field(default_factory=list)
    day_cost_aed: AED = Decimal("0")
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ordered_non_overlapping(self) -> DayPlan:
        for prev, nxt in zip(self.slots, self.slots[1:], strict=False):
            if nxt.start < prev.start:
                raise ValueError(f"slots on {self.date} are not in start-time order")
            if nxt.start < prev.end:
                raise ValueError(
                    f"overlapping slots on {self.date}: {prev.start}-{prev.end} "
                    f"and {nxt.start}-{nxt.end}"
                )
        return self

    @property
    def slot_cost_sum(self) -> Decimal:
        return quantize_aed(sum((s.cost_aed for s in self.slots), Decimal("0")))


class Itinerary(StrictModel):
    """A versioned, day-by-day plan. ``version``/``parent_id`` are set by the repository."""

    id: UUID = Field(default_factory=uuid4)
    session_id: str
    version: int = Field(default=1, ge=1)
    parent_id: UUID | None = None
    mode: PlannerMode
    title: str = ""
    summary: str = ""
    days: list[DayPlan] = Field(min_length=1)
    hotel: HotelOption | None = None
    cost: CostBreakdown
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _days_ordered(self) -> Itinerary:
        dates = [d.date for d in self.days]
        if dates != sorted(set(dates)):
            raise ValueError("itinerary days must have unique dates in ascending order")
        if self.parent_id is not None and self.parent_id == self.id:
            raise ValueError("an itinerary cannot be its own parent")
        return self

    @property
    def start_date(self) -> date:
        return self.days[0].date

    @property
    def end_date(self) -> date:
        return self.days[-1].date

    def venue_ids(self) -> set[str]:
        """All grounded venue ids referenced (for the grounding rule)."""
        ids = {s.activity.ref_id for d in self.days for s in d.slots if s.activity.ref_id}
        if self.hotel is not None:
            ids.add(self.hotel.id)
        return ids
