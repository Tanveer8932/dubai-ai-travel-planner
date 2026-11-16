"""Differences between two saved itinerary versions (Phase 5 sessions page, AC5.4)."""

from __future__ import annotations

import datetime as dt
from datetime import time
from decimal import Decimal

from pydantic import Field

from dubai_planner.models.common import StrictModel


class SlotRef(StrictModel):
    """A slot as the diff sees it: same time, kind and venue ⇒ the same slot."""

    start: time
    end: time
    kind: str
    name: str
    ref_id: str | None = None
    cost_aed: Decimal = Decimal("0")

    def label(self) -> str:
        return f"{self.start:%H:%M}-{self.end:%H:%M} {self.name}"


class DayDiff(StrictModel):
    day_index: int = Field(ge=0)
    date: dt.date | None = None
    added: list[SlotRef] = Field(default_factory=list)
    removed: list[SlotRef] = Field(default_factory=list)
    theme_before: str = ""
    theme_after: str = ""
    cost_before: Decimal = Decimal("0")
    cost_after: Decimal = Decimal("0")

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.theme_before != self.theme_after)


class VersionDiff(StrictModel):
    """``old`` → ``new``: cost delta per category and the slots each day gained or lost."""

    session_id: str
    old_version: int
    new_version: int
    cost_delta: dict[str, Decimal] = Field(
        default_factory=dict, description="category → new - old (AED), incl. 'total'")
    budget_before: Decimal | None = None
    budget_after: Decimal | None = None
    days: list[DayDiff] = Field(default_factory=list)

    @property
    def changed_days(self) -> list[int]:
        return [d.day_index for d in self.days if d.changed]

    @property
    def unchanged_days(self) -> list[int]:
        return [d.day_index for d in self.days if not d.changed]
