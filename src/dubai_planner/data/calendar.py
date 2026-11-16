"""Cultural calendar: Ramadan window, UAE public holidays and recurring events.

Fully offline: Islamic dates come from ``hijridate`` (Umm al-Qura tables) and are
flagged ``approximate`` because the actual start depends on moon sighting.
Nager.Date was evaluated as an online holiday source but does not cover the UAE
(HTTP 204 for ``AE``, checked 2026-11-15), so fixed holidays come from seed data.
"""

from __future__ import annotations

from datetime import date, timedelta

from hijridate import Gregorian, Hijri

from dubai_planner.data.seed import load_catalog
from dubai_planner.models.common import DataKind, Provenance
from dubai_planner.models.context import EventCategory, EventInfo, RecurringEvent

HIJRI_NOTE = (
    "Computed with hijridate (Umm al-Qura); the official date depends on moon sighting "
    "and may differ by 1-2 days."
)


def _hijri_provenance() -> Provenance:
    return Provenance(
        kind=DataKind.ESTIMATED,
        source="hijridate (Umm al-Qura calendar)",
        as_of=date.today(),
        note=HIJRI_NOTE,
    )


def _hijri_years_overlapping(start: date, end: date) -> range:
    first = Gregorian(start.year, start.month, start.day).to_hijri().year
    last = Gregorian(end.year, end.month, end.day).to_hijri().year
    return range(first - 1, last + 2)


def _to_date(h: Hijri) -> date:
    g = h.to_gregorian()
    return date(g.year, g.month, g.day)


def ramadan_window(hijri_year: int) -> tuple[date, date]:
    """First and last Gregorian day of Ramadan (Hijri month 9) for ``hijri_year``."""
    first = Hijri(hijri_year, 9, 1)
    start = _to_date(first)
    return start, start + timedelta(days=first.month_length() - 1)


def ramadan_overlap(start: date, end: date) -> EventInfo | None:
    """Ramadan period overlapping ``[start, end]``, if any."""
    for hy in _hijri_years_overlapping(start, end):
        r_start, r_end = ramadan_window(hy)
        if r_start <= end and start <= r_end:
            return EventInfo(
                id=f"cal_ramadan_{hy}",
                name=f"Ramadan {hy} AH",
                category=EventCategory.RELIGIOUS,
                start_date=r_start,
                end_date=r_end,
                description=(
                    "Holy month: no eating or drinking in public during daylight; many "
                    "restaurants open after sunset (iftar); some opening hours change."
                ),
                approximate=True,
                provenance=_hijri_provenance(),
            )
    return None


def public_holidays(start: date, end: date) -> list[EventInfo]:
    seed = load_catalog().holidays
    out: list[EventInfo] = []
    for year in range(start.year, end.year + 1):
        for fh in seed.fixed:
            day = date(year, fh.month, fh.day)
            if start <= day <= end:
                out.append(EventInfo(
                    id=f"{fh.id}_{year}", name=fh.name, category=EventCategory.PUBLIC_HOLIDAY,
                    start_date=day, end_date=day, provenance=fh.provenance,
                ))
    for hy in _hijri_years_overlapping(start, end):
        for hh in seed.hijri:
            h_start = _to_date(Hijri(hy, hh.hijri_month, hh.hijri_day))
            h_end = h_start + timedelta(days=hh.duration_days - 1)
            if h_start <= end and start <= h_end:
                out.append(EventInfo(
                    id=f"{hh.id}_{hy}", name=hh.name, category=EventCategory.PUBLIC_HOLIDAY,
                    start_date=h_start, end_date=h_end, approximate=True,
                    provenance=hh.provenance,
                ))
    return sorted(out, key=lambda e: e.start_date)


def _occurrences(event: RecurringEvent, start: date, end: date) -> list[EventInfo]:
    out = []
    for year in range(start.year - 1, end.year + 1):
        if event.odd_years_only and year % 2 == 0:
            continue
        ev_start = date(year, event.start.month, event.start.day)
        spans_new_year = (event.end.month, event.end.day) < (event.start.month, event.start.day)
        ev_end = date(year + 1 if spans_new_year else year, event.end.month, event.end.day)
        if ev_start <= end and start <= ev_end:
            out.append(EventInfo(
                id=f"{event.id}_{year}", name=event.name, category=event.category,
                start_date=ev_start, end_date=ev_end, area=event.area,
                description=event.description, approximate=True, provenance=event.provenance,
            ))
    return out


def get_events(start: date, end: date) -> list[EventInfo]:
    """Recurring events, public holidays and Ramadan overlapping the trip dates."""
    if end < start:
        raise ValueError("end must not be before start")
    events: list[EventInfo] = []
    for ev in load_catalog().events:
        events.extend(_occurrences(ev, start, end))
    events.extend(public_holidays(start, end))
    ramadan = ramadan_overlap(start, end)
    if ramadan is not None:
        events.append(ramadan)
    return sorted(events, key=lambda e: (e.start_date, e.id))
