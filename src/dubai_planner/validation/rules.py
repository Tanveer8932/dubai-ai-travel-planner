"""Deterministic itinerary rules (``architecture.md §7``), shared by both orchestrators.

Each rule is a pure function ``(RuleContext) -> list[RuleViolation]``. They never call an
LLM or the network: weather comes from the research bundle (or seed climatology),
Ramadan from ``hijridate``, venue facts from tool results / seed data.

Rules are written defensively: they also check invariants the Pydantic models already
enforce (e.g. no overlapping slots), so an itinerary built with ``model_construct`` or
deserialised from an older version is still reported on rather than trusted.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from dubai_planner.config import get_settings
from dubai_planner.data import calendar
from dubai_planner.data.seed import SeedCatalog, load_catalog
from dubai_planner.models.common import quantize_aed
from dubai_planner.models.itinerary import ActivityKind, DayPlan, Itinerary, TimeSlot
from dubai_planner.models.options import AttractionOption, DiningOption, HotelOption, Weekday
from dubai_planner.models.request import AccessibilityNeed, TripRequest
from dubai_planner.models.research import ResearchBundle
from dubai_planner.models.review import RuleViolation, Severity
from dubai_planner.tools.costs import day_cost
from dubai_planner.validation.thresholds import (
    HEAT_LIMIT_C,
    HOT_MONTHS,
    MAX_LEG_MINUTES,
    MIDDAY_END,
    MIDDAY_START,
    PACE_MAX_ACTIVITIES,
)

Venue = AttractionOption | HotelOption | DiningOption
ZERO = Decimal("0")
SIGHTSEEING = frozenset({ActivityKind.ATTRACTION, ActivityKind.EVENT})


@dataclass
class RuleContext:
    itinerary: Itinerary
    request: TripRequest
    research: ResearchBundle | None = None
    catalog: SeedCatalog = field(default_factory=load_catalog)
    tolerance: Decimal = field(default_factory=lambda: get_settings().budget_tolerance)

    def venue(self, ref_id: str | None) -> Venue | None:
        """Resolve an id from tool results first, then from seed data."""
        if not ref_id:
            return None
        if self.research is not None:
            pool: list[Venue] = [*self.research.attractions, *self.research.hotels,
                                 *self.research.dining]
            for venue in pool:
                if venue.id == ref_id:
                    return venue
        return self.catalog.venue(ref_id)

    def max_temp(self, day: date) -> float:
        weather = self.research.weather if self.research is not None else None
        daily = weather.for_date(day) if weather is not None else None
        if daily is not None:
            return daily.temp_max_c
        return self.catalog.climate.for_month(day.month).avg_high_c

    def is_day_trip(self, ref_id: str | None) -> bool:
        venue = self.venue(ref_id)
        if isinstance(venue, AttractionOption) and venue.day_trip:
            return True
        area = self.catalog.area(venue.area) if venue is not None else None
        return bool(area and area.day_trip)


def _v(rule_id: str, severity: Severity, message: str, *, day: int | None = None,
       slot: int | None = None, **details: object) -> RuleViolation:
    return RuleViolation(rule_id=rule_id, severity=severity, message=message, day_index=day,
                         slot_index=slot, details={k: str(v) for k, v in details.items()})


def _slots(ctx: RuleContext) -> list[tuple[int, DayPlan, int, TimeSlot]]:
    return [(d, day, s, slot) for d, day in enumerate(ctx.itinerary.days)
            for s, slot in enumerate(day.slots)]


# --------------------------------------------------------------------------- budget


def budget_total(ctx: RuleContext) -> list[RuleViolation]:
    budget = ctx.request.total_budget_aed()
    limit = quantize_aed(budget * (1 + ctx.tolerance))
    total = ctx.itinerary.cost.total
    if total <= limit:
        return []
    over = quantize_aed(total - budget)
    pct = (over / budget * 100).quantize(Decimal("0.1"))
    return [_v("budget.total", Severity.ERROR,
               f"Estimated total AED {total:,.2f} exceeds the budget AED {budget:,.2f} by "
               f"AED {over:,.2f} ({pct}%), above the {ctx.tolerance * 100:.0f}% tolerance.",
               total=total, budget=budget, limit=limit)]


def budget_arithmetic(ctx: RuleContext) -> list[RuleViolation]:
    cost = ctx.itinerary.cost
    out: list[RuleViolation] = []
    if cost.category_sum != cost.total:
        out.append(_v("budget.arithmetic", Severity.ERROR,
                      f"Category amounts sum to AED {cost.category_sum} but total is "
                      f"AED {cost.total}.", category_sum=cost.category_sum, total=cost.total))
    if cost.items:
        item_sum = quantize_aed(sum((i.amount_aed for i in cost.items), ZERO))
        if item_sum != cost.total:
            out.append(_v("budget.arithmetic", Severity.ERROR,
                          f"Cost items sum to AED {item_sum} but total is AED {cost.total}.",
                          item_sum=item_sum, total=cost.total))
    for d, day in enumerate(ctx.itinerary.days):
        expected = day_cost(day)
        if day.day_cost_aed != expected:
            out.append(_v("budget.arithmetic", Severity.ERROR,
                          f"Day {d + 1} cost is AED {day.day_cost_aed} but its slots and legs "
                          f"add up to AED {expected}.", day=d, stated=day.day_cost_aed,
                          expected=expected))
    days_sum = quantize_aed(sum((d.day_cost_aed for d in ctx.itinerary.days), ZERO))
    day_categories = quantize_aed(cost.activities + cost.food + cost.transport)
    if days_sum != day_categories:
        out.append(_v("budget.arithmetic", Severity.ERROR,
                      f"Day costs sum to AED {days_sum} but activities + food + transport is "
                      f"AED {day_categories}.", days_sum=days_sum, categories=day_categories))
    return out


# --------------------------------------------------------------------------- time


def time_overlap(ctx: RuleContext) -> list[RuleViolation]:
    out = []
    for d, day in enumerate(ctx.itinerary.days):
        ordered = sorted(enumerate(day.slots), key=lambda pair: pair[1].start)
        for (_, prev), (j, nxt) in itertools.pairwise(ordered):
            if nxt.start < prev.end:
                out.append(_v("time.overlap", Severity.ERROR,
                              f"Day {d + 1}: '{nxt.activity.name}' ({nxt.start:%H:%M}) starts "
                              f"before '{prev.activity.name}' ends ({prev.end:%H:%M}).",
                              day=d, slot=j))
    return out


def time_opening_hours(ctx: RuleContext) -> list[RuleViolation]:
    out = []
    for d, day, s, slot in _slots(ctx):
        venue = ctx.venue(slot.activity.ref_id)
        if venue is None or slot.activity.kind is ActivityKind.HOTEL:
            continue
        if isinstance(venue, AttractionOption) and not venue.open_in_month(day.date.month):
            out.append(_v("time.opening_hours", Severity.WARNING,
                          f"{venue.name} is seasonal and normally closed in "
                          f"{day.date:%B}.", day=d, slot=s))
            continue
        hours = venue.opening_hours
        if hours is not None and not hours.is_open(Weekday(day.date.weekday()), slot.start,
                                                   slot.end):
            out.append(_v("time.opening_hours", Severity.WARNING,
                          f"{venue.name} is usually open {hours.open:%H:%M}-{hours.close:%H:%M}"
                          f" (closed: {[w.name for w in hours.closed_days] or 'none'}); slot is "
                          f"{day.date:%a} {slot.start:%H:%M}-{slot.end:%H:%M}.", day=d, slot=s))
    return out


# --------------------------------------------------------------------------- heat / geo / pace


def heat_midday_outdoor(ctx: RuleContext) -> list[RuleViolation]:
    out = []
    for d, day, s, slot in _slots(ctx):
        if slot.indoor or slot.activity.kind in (ActivityKind.TRANSIT, ActivityKind.HOTEL):
            continue
        if not (slot.start < MIDDAY_END and slot.end > MIDDAY_START):
            continue
        tmax = ctx.max_temp(day.date)
        if tmax <= HEAT_LIMIT_C:
            continue
        severity = Severity.ERROR if day.date.month in HOT_MONTHS else Severity.WARNING
        out.append(_v("heat.midday_outdoor", severity,
                      f"Day {d + 1}: outdoor '{slot.activity.name}' at {slot.start:%H:%M}-"
                      f"{slot.end:%H:%M} with a forecast/typical high of {tmax:.0f} °C; move it "
                      f"before 11:00 or after 16:00, or swap for an indoor activity.",
                      day=d, slot=s, temp_max_c=tmax))
    return out


def geo_travel_time(ctx: RuleContext) -> list[RuleViolation]:
    out = []
    for d, _day, s, slot in _slots(ctx):
        leg = slot.transport_from_prev
        if leg is None or leg.duration_min <= MAX_LEG_MINUTES:
            continue
        if ctx.is_day_trip(slot.activity.ref_id) or ctx.is_day_trip(leg.from_ref):
            continue
        out.append(_v("geo.travel_time", Severity.WARNING,
                      f"Day {d + 1}: {leg.duration_min} min by {leg.mode.value} to "
                      f"'{slot.activity.name}' (limit {MAX_LEG_MINUTES} min outside day trips).",
                      day=d, slot=s, minutes=leg.duration_min))
    return out


def pace_density(ctx: RuleContext) -> list[RuleViolation]:
    limit = PACE_MAX_ACTIVITIES[ctx.request.pace]
    out = []
    for d, day in enumerate(ctx.itinerary.days):
        count = sum(1 for slot in day.slots if slot.activity.kind in SIGHTSEEING)
        if count > limit:
            out.append(_v("pace.density", Severity.WARNING,
                          f"Day {d + 1} has {count} activities; a {ctx.request.pace.value} pace "
                          f"allows at most {limit}.", day=d, count=count, limit=limit))
    return out


# --------------------------------------------------------------------------- constraints


def constraints_dietary(ctx: RuleContext) -> list[RuleViolation]:
    needs = list(ctx.request.dietary)
    if not needs:
        return []
    out = []
    for d, _day, s, slot in _slots(ctx):
        if slot.activity.kind is not ActivityKind.DINING:
            continue
        venue = ctx.venue(slot.activity.ref_id)
        if isinstance(venue, DiningOption) and not venue.caters_for(needs):
            missing = [n.value for n in needs if n not in venue.dietary]
            out.append(_v("constraints.dietary", Severity.ERROR,
                          f"Day {d + 1}: {venue.name} is not listed as catering for "
                          f"{', '.join(missing)}.", day=d, slot=s, missing=missing))
    return out


def constraints_accessibility(ctx: RuleContext) -> list[RuleViolation]:
    if AccessibilityNeed.WHEELCHAIR not in ctx.request.accessibility:
        return []
    out = []
    seen: set[str] = set()
    refs: list[tuple[int | None, int | None, str | None]] = [
        (d, s, slot.activity.ref_id) for d, _day, s, slot in _slots(ctx)]
    if ctx.itinerary.hotel is not None:
        refs.append((None, None, ctx.itinerary.hotel.id))
    for d, s, ref in refs:
        venue = ctx.venue(ref)
        if venue is None or venue.id in seen:
            continue
        seen.add(venue.id)
        if venue.wheelchair_accessible is False:
            out.append(_v("constraints.accessibility", Severity.WARNING,
                          f"{venue.name} is recorded as not wheelchair accessible.",
                          day=d, slot=s))
        elif venue.wheelchair_accessible is None:
            out.append(_v("constraints.accessibility", Severity.INFO,
                          f"Wheelchair access at {venue.name} is unknown; check before visiting.",
                          day=d, slot=s))
    return out


def culture_ramadan(ctx: RuleContext) -> list[RuleViolation]:
    ramadan = calendar.ramadan_overlap(ctx.itinerary.start_date, ctx.itinerary.end_date)
    if ramadan is None:
        return []
    return [_v("culture.ramadan", Severity.INFO,
               f"The trip overlaps Ramadan (approx. {ramadan.start_date:%d %b} - "
               f"{ramadan.end_date:%d %b %Y}): avoid eating, drinking or smoking in public "
               "during daylight, dress modestly, expect changed opening hours, and consider "
               "an iftar after sunset.", start=ramadan.start_date, end=ramadan.end_date)]


def grounding_known_entities(ctx: RuleContext) -> list[RuleViolation]:
    known = ctx.research.known_ids() if ctx.research is not None else ctx.catalog.known_ids()
    out = []
    for d, _day, s, slot in _slots(ctx):
        ref = slot.activity.ref_id
        if ref and ref not in known:
            out.append(_v("grounding.known_entities", Severity.WARNING,
                          f"Day {d + 1}: '{slot.activity.name}' ({ref}) is not in the "
                          "researched tool results or seed data.", day=d, slot=s, ref_id=ref))
    hotel = ctx.itinerary.hotel
    if hotel is not None and hotel.id not in known:
        out.append(_v("grounding.known_entities", Severity.WARNING,
                      f"Hotel {hotel.name} ({hotel.id}) is not in the researched results.",
                      ref_id=hotel.id))
    return out


Rule = Callable[[RuleContext], list[RuleViolation]]

RULES: dict[str, Rule] = {
    "budget.total": budget_total,
    "budget.arithmetic": budget_arithmetic,
    "time.overlap": time_overlap,
    "time.opening_hours": time_opening_hours,
    "heat.midday_outdoor": heat_midday_outdoor,
    "geo.travel_time": geo_travel_time,
    "pace.density": pace_density,
    "constraints.dietary": constraints_dietary,
    "constraints.accessibility": constraints_accessibility,
    "culture.ramadan": culture_ramadan,
    "grounding.known_entities": grounding_known_entities,
}

_SEVERITY_ORDER = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}


def validate_itinerary(
    itinerary: Itinerary,
    request: TripRequest,
    research: ResearchBundle | None = None,
    *,
    catalog: SeedCatalog | None = None,
    tolerance: Decimal | None = None,
    rules: dict[str, Rule] | None = None,
) -> list[RuleViolation]:
    """Run every rule; violations sorted errors → warnings → info, then by position."""
    ctx = RuleContext(itinerary=itinerary, request=request, research=research,
                      catalog=catalog or load_catalog(),
                      tolerance=get_settings().budget_tolerance if tolerance is None
                      else tolerance)
    found = [v for rule in (rules or RULES).values() for v in rule(ctx)]
    return sorted(found, key=lambda v: (_SEVERITY_ORDER[v.severity], v.day_index or 0,
                                        v.slot_index or 0, v.rule_id))


def has_errors(violations: list[RuleViolation]) -> bool:
    return any(v.severity is Severity.ERROR for v in violations)
