"""Benchmark scenarios (implementation plan, Phase 7 task 1).

A scenario is one conversation (one or more user turns) plus the properties the final
plan must have. Expectations are written from the *user's words*, not from what the
planner parsed, so a parsing mistake (e.g. a missed "vegetarian") counts against the
mode instead of silently lowering the bar.

Scenarios live in ``eval/scenarios/*.json`` (one file each) and are validated here.
"""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from dubai_planner.config import PROJECT_ROOT
from dubai_planner.models.common import StrictModel
from dubai_planner.models.request import AccessibilityNeed, DietaryFlag, Pace

SCENARIO_DIR = PROJECT_ROOT / "eval" / "scenarios"
MIN_SCENARIOS = 12


class ExpectedOutcome(StrEnum):
    """How the conversation should end."""

    APPROVED = "approved"  # a plan within budget, approved by the review loop
    NEEDS_CLARIFICATION = "needs_clarification"  # the planner must ask a question
    INFEASIBLE_EXPLAINED = "infeasible_explained"  # over budget, *with* an explanation


class Expectations(StrictModel):
    """Hard constraints and expected properties of the final plan (all optional)."""

    outcome: ExpectedOutcome = ExpectedOutcome.APPROVED
    asks_clarification: bool = Field(
        default=False, description="The first turn must end with a clarification question")
    days: int | None = Field(default=None, ge=1)
    adults: int | None = Field(default=None, ge=1)
    children: int | None = Field(default=None, ge=0)
    budget_aed: Decimal | None = Field(default=None, gt=0,
                                       description="Expected total budget after all turns")
    dietary: list[DietaryFlag] = Field(default_factory=list)
    accessibility: list[AccessibilityNeed] = Field(default_factory=list)
    pace: Pace | None = None
    interests: list[str] = Field(default_factory=list,
                                 description="Preference alignment (soft metric)")
    heat_safe: bool = Field(default=False,
                            description="No outdoor slot 11:00-16:00 on any day")
    family_friendly: bool = False
    latest_end: time | None = Field(default=None, description="'no late nights'")
    day_trip_area: str | None = Field(default=None,
                                      description="An attraction in this area, e.g. abu_dhabi")
    indoor_days: list[int] = Field(default_factory=list,
                                   description="1-based days whose attractions are all indoor")


class Scenario(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    category: str
    turns: list[str] = Field(min_length=1, max_length=6)
    today: date = date(2026, 11, 16)
    expect: Expectations = Field(default_factory=Expectations)
    notes: str = ""

    @model_validator(mode="after")
    def _turns_not_blank(self) -> Scenario:
        if any(not t.strip() for t in self.turns):
            raise ValueError("scenario turns must not be blank")
        return self

    @property
    def multi_turn(self) -> bool:
        return len(self.turns) > 1


def load_scenarios(directory: Path | None = None, ids: list[str] | None = None
                   ) -> list[Scenario]:
    """All scenarios in ``directory`` (sorted by file name), optionally filtered by id."""
    folder = directory or SCENARIO_DIR
    found = [Scenario.model_validate_json(p.read_text(encoding="utf-8"))
             for p in sorted(folder.glob("*.json"))]
    seen: set[str] = set()
    for s in found:
        if s.id in seen:
            raise ValueError(f"duplicate scenario id {s.id!r}")
        seen.add(s.id)
    if ids:
        unknown = sorted(set(ids) - seen)
        if unknown:
            raise ValueError(f"unknown scenario id(s): {', '.join(unknown)}")
        found = [s for s in found if s.id in ids]
    return found
