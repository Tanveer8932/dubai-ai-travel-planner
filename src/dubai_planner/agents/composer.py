"""Itinerary composer: the LLM drafts days/slots by id; Python assembles the Itinerary.

Division of labour:

- **LLM** (:class:`DraftItinerary`, a small schema): which researched venue goes on which
  day and when, day themes, title and summary. It may only reference ids from the
  context (hotels, attractions, dining, events).
- **Python** (:func:`assemble`): grounds every id against the research bundle, fills
  names/locations/indoor flags from tool data, estimates every transport leg, prices
  every slot for the party and builds the :class:`CostBreakdown` (hard rule 4). Any
  problem (unknown id, bad time, overlap, missing day) raises ``ValueError`` and is sent
  back to the model as the repair message.

:func:`baseline_draft` is a deterministic, heat- and pace-aware scheduler. The mock
provider uses it as its reply, and ``run(..., fallback=True)`` uses it when the model
still fails after the repair attempt.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from datetime import date, time, timedelta
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, model_validator

from dubai_planner.agents.base import LLMOutput, context_block, load_prompt
from dubai_planner.agents.transport import prefers_taxi
from dubai_planner.llm.errors import StructuredOutputError
from dubai_planner.llm.mock import MockRequest, register_generator
from dubai_planner.llm.structured import build_messages, invoke_structured
from dubai_planner.models.budget import BudgetCategory, BudgetPlan
from dubai_planner.models.common import GeoPoint, StrictModel, quantize_aed
from dubai_planner.models.itinerary import (
    Activity,
    ActivityKind,
    DayPlan,
    Itinerary,
    PlannerMode,
    TimeSlot,
)
from dubai_planner.models.options import AttractionOption, DiningOption, HotelOption, MealType
from dubai_planner.models.request import TripRequest
from dubai_planner.models.research import ResearchBundle
from dubai_planner.models.trace import TraceEventType
from dubai_planner.models.transport import TransportMode
from dubai_planner.tools.costs import day_cost, party_cost
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import agent_span, emit
from dubai_planner.validation.thresholds import (
    HEAT_LIMIT_C,
    MIDDAY_END,
    MIDDAY_START,
    PACE_MAX_ACTIVITIES,
    PACE_TARGET_ACTIVITIES,
)

AGENT = "composer"
_TIME = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*$")
SlotKind = Literal["attraction", "dining", "hotel", "free_time", "event"]


# --------------------------------------------------------------------------- I/O models


class ComposerInput(StrictModel):
    session_id: str = Field(min_length=1)
    mode: PlannerMode
    request: TripRequest
    research: ResearchBundle
    budget: BudgetPlan
    feedback: list[str] = Field(default_factory=list, description="Reviewer/rule feedback")
    previous: Itinerary | None = None
    locked_days: list[int] = Field(default_factory=list, description="0-based days to keep")

    @model_validator(mode="after")
    def _locked_days_valid(self) -> ComposerInput:
        if not self.locked_days:
            return self
        if self.previous is None:
            raise ValueError("locked_days requires the previous itinerary")
        limit = min(len(self.previous.days), self.request.num_days)
        bad = sorted(d for d in self.locked_days if not 0 <= d < limit)
        if bad:
            raise ValueError(f"locked_days {bad} are outside the previous itinerary")
        return self


class DraftSlot(LLMOutput):
    start: str = Field(description="HH:MM, 24-hour")
    end: str = Field(description="HH:MM, 24-hour, after start, same day")
    kind: SlotKind
    ref_id: str | None = Field(default=None, description="Id from the context lists")
    title: str | None = Field(default=None, description="Only for free_time / event")
    note: str = Field(default="", max_length=300)


class DraftDay(LLMOutput):
    day: int = Field(ge=1, description="1-based day number")
    theme: str = Field(default="", max_length=120)
    slots: list[DraftSlot] = Field(default_factory=list)


class DraftItinerary(LLMOutput):
    title: str = Field(default="", max_length=120)
    summary: str = Field(default="", max_length=1200)
    hotel_id: str | None = None
    days: list[DraftDay] = Field(min_length=1)


class AttractionBrief(StrictModel):
    id: str
    name: str
    area: str
    lat: float
    lon: float
    indoor: bool
    duration_min: int
    best_time: list[str]
    day_trip: bool
    opening: str | None
    closed_days: list[str]
    cost_aed: str


class DiningBrief(StrictModel):
    id: str
    name: str
    area: str
    lat: float
    lon: float
    meals: list[str]
    cost_aed: str


class HotelBrief(StrictModel):
    id: str
    name: str
    area: str
    lat: float
    lon: float


class DayBrief(StrictModel):
    day: int
    date: date
    weekday: str
    max_temp_c: float
    hot: bool


class ComposerContext(StrictModel):
    trip: dict[str, Any]
    max_activities_per_day: int
    target_activities_per_day: int
    midday_window: str
    days: list[DayBrief]
    hotels: list[HotelBrief]
    attractions: list[AttractionBrief]
    dining: list[DiningBrief]
    events: list[dict[str, str]]
    budget: dict[str, str]
    locked_days: list[int]
    feedback: list[str]


# --------------------------------------------------------------------------- context


def _max_temp(research: ResearchBundle, tools: Toolbox, day: date) -> float:
    daily = research.weather.for_date(day) if research.weather is not None else None
    return daily.temp_max_c if daily else tools.catalog.climate.for_month(day.month).avg_high_c


def affordable_activities(plan: BudgetPlan) -> Decimal:
    """What is left for paid activities once the other categories and contingency fit."""
    fixed = sum((plan.allocation(c) for c in (BudgetCategory.ACCOMMODATION, BudgetCategory.FOOD,
                                              BudgetCategory.TRANSPORT)), Decimal("0"))
    cap = plan.budget_aed / (1 + plan.contingency_pct) - fixed
    return quantize_aed(max(Decimal("0"), min(cap, plan.allocation(BudgetCategory.ACTIVITIES))))


def build_context(data: ComposerInput, tools: Toolbox) -> ComposerContext:
    req, research = data.request, data.research
    party = req.travellers
    days = []
    for i in range(req.num_days):
        day = req.start_date + timedelta(days=i)
        tmax = _max_temp(research, tools, day)
        days.append(DayBrief(day=i + 1, date=day, weekday=day.strftime("%a"),
                             max_temp_c=round(tmax, 1), hot=tmax > HEAT_LIMIT_C))
    attractions = [
        AttractionBrief(
            id=a.id, name=a.name, area=a.area, lat=a.location.lat, lon=a.location.lon,
            indoor=a.indoor, duration_min=a.typical_duration_min,
            best_time=[t.value for t in a.best_time], day_trip=a.day_trip,
            opening=(f"{a.opening_hours.open:%H:%M}-{a.opening_hours.close:%H:%M}"
                     if a.opening_hours else None),
            closed_days=[w.name for w in a.opening_hours.closed_days] if a.opening_hours else [],
            cost_aed=str(party_cost(a.price, party)),
        )
        for a in research.attractions
    ]
    dining = [
        DiningBrief(id=d.id, name=d.name, area=d.area, lat=d.location.lat, lon=d.location.lon,
                    meals=[m.value for m in d.meal_types],
                    cost_aed=str(party_cost(d.avg_spend, party)))
        for d in research.dining
    ]
    return ComposerContext(
        trip={"start_date": req.start_date.isoformat(), "days": req.num_days,
              "adults": party.adults, "children_ages": party.children_ages,
              "pace": req.pace.value, "interests": req.interests,
              "dietary": [d.value for d in req.dietary], "must_include": req.must_include},
        max_activities_per_day=PACE_MAX_ACTIVITIES[req.pace],
        target_activities_per_day=PACE_TARGET_ACTIVITIES[req.pace],
        midday_window=f"{MIDDAY_START:%H:%M}-{MIDDAY_END:%H:%M}",
        days=days,
        hotels=[HotelBrief(id=h.id, name=h.name, area=h.area, lat=h.location.lat,
                           lon=h.location.lon) for h in research.hotels[:3]],
        attractions=attractions,
        dining=dining,
        events=[{"id": e.id, "name": e.name,
                 "dates": f"{e.start_date.isoformat()} to {e.end_date.isoformat()}"}
                for e in research.events],
        budget={"total": str(data.budget.budget_aed),
                "activities": str(data.budget.allocation(BudgetCategory.ACTIVITIES)),
                "activities_cap": str(affordable_activities(data.budget)),
                "food": str(data.budget.allocation(BudgetCategory.FOOD))},
        locked_days=[d + 1 for d in data.locked_days],
        feedback=data.feedback,
    )


# --------------------------------------------------------------------------- assembly


def parse_time(value: str, field: str) -> time:
    match = _TIME.match(value)
    if not match or int(match.group(1)) > 23 or int(match.group(2)) > 59:
        raise ValueError(f"{field} {value!r} is not a valid HH:MM time")
    return time(int(match.group(1)), int(match.group(2)))


def assemble(draft: DraftItinerary, data: ComposerInput, tools: Toolbox) -> Itinerary:
    """Ground, price and validate a draft. Raises ``ValueError`` describing every problem."""
    req, research = data.request, data.research
    party = req.travellers
    hotels = {h.id: h for h in research.hotels}
    attractions = {a.id: a for a in research.attractions}
    dining = {d.id: d for d in research.dining}
    events = {e.id: e for e in research.events}
    problems: list[str] = []

    hotel: HotelOption | None = None
    if draft.hotel_id:
        hotel = hotels.get(draft.hotel_id)
        if hotel is None:
            problems.append(f"hotel_id {draft.hotel_id!r} is not one of {sorted(hotels)}")
    elif research.hotels and req.nights > 0:
        hotel = research.hotels[0]

    numbers = [d.day for d in draft.days]
    expected = set(range(1, req.num_days + 1))
    dupes = sorted({n for n in numbers if numbers.count(n) > 1})
    if dupes:
        problems.append(f"day numbers repeated: {dupes}")
    locked = {d + 1 for d in data.locked_days}
    missing = sorted(expected - set(numbers) - locked)
    extra = sorted(set(numbers) - expected)
    if missing:
        problems.append(f"missing day(s) {missing}; plan every day 1..{req.num_days}")
    if extra:
        problems.append(f"day(s) {extra} are outside the trip (1..{req.num_days})")

    taxi_first = prefers_taxi(req)
    used_attractions: Counter[str] = Counter()
    day_plans: dict[int, DayPlan] = {}
    for draft_day in draft.days:
        n = draft_day.day
        if n not in expected or n in locked:
            continue
        day_date = req.start_date + timedelta(days=n - 1)
        slots: list[TimeSlot] = []
        prev_point: GeoPoint | None = hotel.location if hotel else None
        prev_ref: str | None = hotel.id if hotel else None
        timed: list[tuple[time, time, DraftSlot]] = []
        for i, ds in enumerate(draft_day.slots):
            label = f"day {n} slot {i + 1}"
            try:
                start, end = parse_time(ds.start, f"{label} start"), parse_time(ds.end,
                                                                                 f"{label} end")
            except ValueError as exc:
                problems.append(str(exc))
                continue
            if end <= start:
                problems.append(f"{label}: end {ds.end} must be after start {ds.start}")
                continue
            timed.append((start, end, ds))
        timed.sort(key=lambda t: t[0])
        for (_s1, e1, a), (s2, _e2, b) in itertools.pairwise(timed):
            if s2 < e1:
                problems.append(f"day {n}: {a.ref_id or a.title} ({a.start}-{a.end}) overlaps "
                                f"{b.ref_id or b.title} ({b.start}-{b.end})")
        for start, end, ds in timed:
            venue: AttractionOption | DiningOption | HotelOption | None = None
            kind = ActivityKind(ds.kind)
            if kind is ActivityKind.ATTRACTION:
                venue = attractions.get(ds.ref_id or "")
                if venue is not None:
                    used_attractions[venue.id] += 1
            elif kind is ActivityKind.DINING:
                venue = dining.get(ds.ref_id or "")
            elif kind is ActivityKind.HOTEL:
                venue = hotel if hotel is not None and ds.ref_id in (None, hotel.id) else None
            if kind in (ActivityKind.ATTRACTION, ActivityKind.DINING, ActivityKind.HOTEL):
                if venue is None:
                    problems.append(f"day {n}: {ds.kind} ref_id {ds.ref_id!r} is not in the "
                                    f"provided {ds.kind} list")
                    continue
                if isinstance(venue, AttractionOption):
                    slot_cost = party_cost(venue.price, party)
                elif isinstance(venue, DiningOption):
                    slot_cost = party_cost(venue.avg_spend, party)
                else:
                    slot_cost = Decimal("0")
                indoor = venue.indoor if isinstance(venue, AttractionOption) else True
                leg = None
                if prev_point is not None and prev_ref != venue.id:
                    pickup = isinstance(venue, AttractionOption) and "tour pickup" in venue.tags
                    leg = tools.estimate_travel(
                        prev_point, venue.location, travellers=party, prefer_taxi=taxi_first,
                        mode=TransportMode.TOUR_PICKUP if pickup else None,
                        from_ref=prev_ref, to_ref=venue.id)
                prev_point, prev_ref = venue.location, venue.id
                activity = Activity(kind=kind, name=venue.name, ref_id=venue.id,
                                    area=venue.area, location=venue.location,
                                    description=ds.note)
                slots.append(TimeSlot(start=start, end=end, activity=activity,
                                      transport_from_prev=leg, cost_aed=slot_cost,
                                      indoor=indoor))
            else:
                event = events.get(ds.ref_id or "") if kind is ActivityKind.EVENT else None
                if kind is ActivityKind.EVENT and ds.ref_id and event is None:
                    problems.append(f"day {n}: event ref_id {ds.ref_id!r} is not a listed event")
                    continue
                name = (event.name if event else ds.title) or "Free time"
                slots.append(TimeSlot(
                    start=start, end=end,
                    activity=Activity(kind=kind, name=name, ref_id=event.id if event else None,
                                      description=ds.note),
                    cost_aed=Decimal("0"), indoor=kind is ActivityKind.FREE_TIME))
        try:
            plan = DayPlan(date=day_date, theme=draft_day.theme, slots=slots)
        except ValueError as exc:  # overlap already reported above
            if not problems:
                problems.append(f"day {n}: {exc}")
            continue
        day_plans[n] = plan.model_copy(update={"day_cost_aed": day_cost(plan)})

    repeats = sorted(k for k, v in used_attractions.items() if v > 1)
    if repeats:
        problems.append(f"attractions used more than once: {repeats}")
    if problems:
        raise ValueError("; ".join(problems[:12]))

    if data.previous is not None:
        for d in locked:
            if 0 <= d - 1 < len(data.previous.days):
                day_plans[d] = data.previous.days[d - 1]
    days = [day_plans[n] for n in sorted(day_plans)]
    cost = tools.calculate_costs(days, hotel=hotel, request=req,
                                 contingency_pct=data.budget.contingency_pct)
    notes = ["All prices are estimates from curated public data (not live); confirm before "
             "booking."]
    if not data.budget.feasible and data.budget.infeasibility_reason:
        notes.append(data.budget.infeasibility_reason)
    notes.extend(f"Event during your trip: {e.name} ({e.start_date:%d %b}-{e.end_date:%d %b})"
                 for e in research.events[:5])
    return Itinerary(
        session_id=data.session_id, mode=data.mode,
        title=draft.title or f"{req.num_days}-day Dubai itinerary",
        summary=draft.summary, days=days, hotel=hotel, cost=cost, notes=notes,
    )


# --------------------------------------------------------------------------- baseline


def _m(t: time) -> int:
    return t.hour * 60 + t.minute


def _t(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _km(a: tuple[float, float], b: tuple[float, float]) -> float:
    return GeoPoint(lat=a[0], lon=a[1]).distance_km(GeoPoint(lat=b[0], lon=b[1]))


def _open_ok(att: AttractionBrief, weekday: str, start: int, end: int) -> bool:
    if weekday.upper()[:3] in att.closed_days:
        return False
    if att.opening is None:
        return True
    o, c = (parse_time(x, "opening") for x in att.opening.split("-"))
    if _m(c) <= _m(o):  # closes after midnight
        return start >= _m(o) or end <= _m(c)
    return _m(o) <= start and end <= _m(c)


def _period(att: AttractionBrief, hot: bool) -> str:
    evening_only = {"evening", "night"} & set(att.best_time) and not (
        {"morning", "afternoon"} & set(att.best_time))
    if hot and not att.indoor:
        return "morning" if att.duration_min <= 120 and not evening_only else "evening"
    return "evening" if evening_only else "day"


def baseline_draft(ctx: ComposerContext) -> DraftItinerary:
    """Deterministic schedule: cluster by area, heat-aware periods, two meals a day."""
    used: set[str] = set()
    meals_used: Counter[str] = Counter()
    activity_budget = Decimal(ctx.budget.get("activities_cap")
                              or ctx.budget.get("activities") or "0")
    spent = Decimal("0")
    hotel = ctx.hotels[0] if ctx.hotels else None
    locked = set(ctx.locked_days)
    allow_day_trip = len(ctx.days) >= 3
    out_days: list[DraftDay] = []
    for day in ctx.days:
        if day.day in locked:
            out_days.append(DraftDay(day=day.day, theme="(kept from previous version)"))
            continue
        pool = [a for a in ctx.attractions if a.id not in used]
        affordable = [a for a in pool
                      if Decimal(a.cost_aed) == 0 or spent + Decimal(a.cost_aed) <= activity_budget]
        pool = affordable or [a for a in pool if Decimal(a.cost_aed) == 0]
        middle_day = 1 < day.day < len(ctx.days)
        chosen: list[AttractionBrief] = []
        trip = next((a for a in pool if a.day_trip), None)
        if allow_day_trip and middle_day and trip is not None and pool and pool[0].day_trip:
            chosen = [trip]
        else:
            local = [a for a in pool if not a.day_trip]
            if local:
                anchor = local[0]
                rest = sorted(local[1:], key=lambda a: (
                    round(_km((anchor.lat, anchor.lon), (a.lat, a.lon)) / 4),
                    ctx.attractions.index(a)))
                chosen = [anchor, *rest[: ctx.target_activities_per_day - 1]]
        slots = _schedule_day(chosen, day, ctx, meals_used, hotel)
        for s in slots:
            if s.kind == "attraction" and s.ref_id:
                used.add(s.ref_id)
                spent += Decimal(next(a.cost_aed for a in ctx.attractions if a.id == s.ref_id))
        areas = list(dict.fromkeys(a.area.replace("_", " ").title() for a in chosen
                                   if any(s.ref_id == a.id for s in slots)))
        out_days.append(DraftDay(day=day.day, theme=" & ".join(areas[:2]) or "Free day",
                                 slots=slots))
    return DraftItinerary(
        title=f"{len(ctx.days)}-day Dubai itinerary",
        summary=("Days are grouped by area to limit travel; outdoor sightseeing avoids the "
                 f"{ctx.midday_window} heat window on hot days; two meals a day."),
        hotel_id=hotel.id if hotel else None,
        days=out_days,
    )


def _schedule_day(chosen: list[AttractionBrief], day: DayBrief, ctx: ComposerContext,
                  meals_used: Counter[str], hotel: HotelBrief | None) -> list[DraftSlot]:
    morning = [a for a in chosen if _period(a, day.hot) == "morning"]
    evening = [a for a in chosen if _period(a, day.hot) == "evening"]
    daytime = [a for a in chosen if a not in morning and a not in evening]
    slots: list[DraftSlot] = []
    cursor = 9 * 60
    origin = (hotel.lat, hotel.lon) if hotel else None
    last_point = origin

    def place(att: AttractionBrief, start: int) -> int | None:
        if att.opening is not None:  # wait for opening time rather than skip the venue
            opens, closes = (_m(parse_time(x, "opening")) for x in att.opening.split("-"))
            if closes > opens and start < opens:
                start = opens
        end = start + att.duration_min
        if end > 23 * 60 or not _open_ok(att, day.weekday, start, end):
            return None
        if day.hot and not att.indoor and start < _m(MIDDAY_END) and end > _m(MIDDAY_START):
            return None
        slots.append(DraftSlot(start=_t(start), end=_t(end), kind="attraction", ref_id=att.id))
        return end

    def meal(kind: MealType, earliest: int, length: int) -> int:
        nonlocal last_point
        options = [d for d in ctx.dining if kind.value in d.meals]
        if not options:
            return earliest
        anchor = last_point or (options[0].lat, options[0].lon)
        best = min(options, key=lambda d: (meals_used[d.id], round(_km(anchor, (d.lat, d.lon))),
                                           ctx.dining.index(d)))
        meals_used[best.id] += 1
        slots.append(DraftSlot(start=_t(earliest), end=_t(earliest + length), kind="dining",
                               ref_id=best.id))
        last_point = (best.lat, best.lon)
        return earliest + length

    for att in morning:  # hot-day outdoor: finish by 11:00
        start = max(cursor, _m(MIDDAY_START) - att.duration_min, 7 * 60 + 30)
        end = place(att, start)
        if end is not None:
            cursor, last_point = end + 30, (att.lat, att.lon)
    lunch_done = False
    for att in daytime:
        # Lunch first when this attraction would run through lunchtime from late morning.
        if not lunch_done and cursor >= 11 * 60 + 30 and cursor + att.duration_min > 13 * 60 + 30:
            cursor = meal(MealType.LUNCH, max(cursor, 12 * 60 + 30), 60) + 30
            lunch_done = True
        end = place(att, cursor)
        if end is not None:
            cursor, last_point = end + 30, (att.lat, att.lon)
    if not lunch_done and cursor <= 15 * 60:
        cursor = meal(MealType.LUNCH, max(cursor, 12 * 60 + 30), 60) + 30
    if evening:
        cursor = max(cursor, 17 * 60)
    for att in evening:
        end = place(att, cursor)
        if end is not None:
            cursor, last_point = end + 30, (att.lat, att.lon)
    dinner_start = max(cursor, 19 * 60 + 30)
    if dinner_start + 90 <= 23 * 60 + 30:
        meal(MealType.DINNER, dinner_start, 90)
    return sorted(slots, key=lambda s: s.start)


# --------------------------------------------------------------------------- run


def run(data: ComposerInput, llm: Any, tools: Toolbox, *, fallback: bool = False) -> Itinerary:
    with agent_span(AGENT, feedback=len(data.feedback), locked_days=data.locked_days) as span:
        ctx = build_context(data, tools)
        messages = build_messages(
            system=load_prompt("composer.system"),
            user=load_prompt("composer.user", context=context_block(ctx),
                             feedback="\n".join(f"- {f}" for f in data.feedback) or "- none"),
            schema=DraftItinerary, agent=AGENT,
        )
        try:
            result = invoke_structured(llm, messages, DraftItinerary, agent=AGENT,
                                       postprocess=lambda d: assemble(d, data, tools))
            itinerary, calls, used_fallback = result.value, result.calls, False
        except StructuredOutputError:
            if not fallback:
                raise
            emit(AGENT, TraceEventType.INFO, payload={"fallback": "baseline_draft"})
            itinerary = assemble(baseline_draft(ctx), data, tools)
            itinerary = itinerary.model_copy(update={"notes": [
                *itinerary.notes, "Schedule produced by the deterministic planner (the model's "
                "output was invalid)."]})
            calls, used_fallback = 0, True
        span.update(days=len(itinerary.days), total=str(itinerary.cost.total),
                    within_budget=itinerary.cost.within_budget, llm_calls=calls,
                    fallback=used_fallback)
        return itinerary


def _mock_compose(req: MockRequest) -> DraftItinerary:
    return baseline_draft(ComposerContext.model_validate(req.context or {}))


register_generator(DraftItinerary.__name__, _mock_compose)
