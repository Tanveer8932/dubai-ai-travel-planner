"""Weather & events agent (deterministic): weather context, events, advisories.

Weather: Open-Meteo forecast → same dates last year → seed climatology (never fails).
Events: curated recurring events, UAE public holidays and Ramadan (``hijridate``).
Advisories are short planning notes derived by rules (heat, Ramadan, holidays, rain).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import Field, model_validator

from dubai_planner.models.common import StrictModel
from dubai_planner.models.context import (
    EventCategory,
    EventInfo,
    WeatherContext,
    WeatherSource,
)
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import agent_span

AGENT = "weather_events"
RAIN_MM = 5.0


class WeatherEventsInput(StrictModel):
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def _ordered(self) -> WeatherEventsInput:
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        return self


class WeatherEventsResult(StrictModel):
    weather: WeatherContext
    events: list[EventInfo] = Field(default_factory=list)
    hot_days: list[date] = Field(default_factory=list)
    advisories: list[str] = Field(default_factory=list)


def advisories(weather: WeatherContext, events: list[EventInfo]) -> list[str]:
    notes: list[str] = []
    if weather.source is not WeatherSource.FORECAST:
        notes.append(f"Weather is an estimate ({weather.provenance.label()}), not a forecast.")
    if weather.hot_days:
        notes.append(f"Highs above 38 °C on {len(weather.hot_days)} day(s): plan outdoor "
                     "sightseeing before 11:00 or after 16:00 and keep middays indoors.")
    wet = [d.date for d in weather.days if d.precipitation_mm >= RAIN_MM]
    if wet:
        notes.append(f"Rain possible on {', '.join(d.isoformat() for d in wet)}: have an indoor "
                     "backup.")
    for event in events:
        if event.category is EventCategory.RELIGIOUS and event.id.startswith("cal_ramadan"):
            notes.append(f"{event.name} overlaps the trip (approximate dates): no eating or "
                         "drinking in public in daylight; many restaurants open at iftar.")
        elif event.category is EventCategory.PUBLIC_HOLIDAY:
            notes.append(f"Public holiday: {event.name} ({event.start_date.isoformat()}): "
                         "expect crowds and some changed opening hours.")
    return notes


def run(data: WeatherEventsInput, llm: Any = None, tools: Toolbox | None = None
        ) -> WeatherEventsResult:
    """``llm`` is accepted for a uniform agent signature but not used."""
    tb = tools or Toolbox.default()
    with agent_span(AGENT) as span:
        weather = tb.get_weather_context(data.start_date, data.end_date)
        events = tb.get_events(data.start_date, data.end_date)
        result = WeatherEventsResult(weather=weather, events=events, hot_days=weather.hot_days,
                                     advisories=advisories(weather, events))
        span.update(source=weather.source.value, hot_days=len(result.hot_days),
                    events=len(events))
        return result
