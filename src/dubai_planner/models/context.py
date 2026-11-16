"""Weather and calendar context: daily weather, climate normals, events and holidays."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import Field, model_validator

from dubai_planner.models.common import Provenance, StrictModel

#: Above this daily max (°C) outdoor slots between 11:00 and 16:00 are discouraged.
HEAT_THRESHOLD_C = 38.0


class WeatherSource(StrEnum):
    FORECAST = "forecast"  # Open-Meteo forecast (≤ 16 days ahead)
    HISTORICAL = "historical"  # same calendar dates last year (Open-Meteo archive)
    CLIMATOLOGY = "climatology"  # monthly normals from seed (offline fallback)


class DailyWeather(StrictModel):
    date: date
    temp_max_c: float = Field(ge=-10, le=60)
    temp_min_c: float = Field(ge=-10, le=60)
    precipitation_mm: float = Field(default=0.0, ge=0)
    uv_index_max: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _min_le_max(self) -> DailyWeather:
        if self.temp_min_c > self.temp_max_c:
            raise ValueError("temp_min_c must not exceed temp_max_c")
        return self

    @property
    def is_hot(self) -> bool:
        return self.temp_max_c > HEAT_THRESHOLD_C


class WeatherContext(StrictModel):
    source: WeatherSource
    days: list[DailyWeather] = Field(min_length=1)
    provenance: Provenance

    @model_validator(mode="after")
    def _dates_ordered(self) -> WeatherContext:
        dates = [d.date for d in self.days]
        if dates != sorted(set(dates)):
            raise ValueError("weather days must be unique and in date order")
        return self

    @property
    def max_temp_c(self) -> float:
        return max(d.temp_max_c for d in self.days)

    @property
    def hot_days(self) -> list[date]:
        return [d.date for d in self.days if d.is_hot]

    def for_date(self, day: date) -> DailyWeather | None:
        return next((d for d in self.days if d.date == day), None)


class MonthlyClimate(StrictModel):
    month: int = Field(ge=1, le=12)
    avg_high_c: float
    avg_low_c: float
    precipitation_mm: float = Field(ge=0)


class ClimateNormals(StrictModel):
    months: list[MonthlyClimate] = Field(min_length=12, max_length=12)
    provenance: Provenance

    @model_validator(mode="after")
    def _all_months(self) -> ClimateNormals:
        if sorted(m.month for m in self.months) != list(range(1, 13)):
            raise ValueError("climate normals need exactly one entry per month")
        return self

    def for_month(self, month: int) -> MonthlyClimate:
        return next(m for m in self.months if m.month == month)


class EventCategory(StrEnum):
    FESTIVAL = "festival"
    SHOPPING = "shopping"
    SPORT = "sport"
    CULTURE = "culture"
    SEASONAL_ATTRACTION = "seasonal_attraction"
    PUBLIC_HOLIDAY = "public_holiday"
    RELIGIOUS = "religious"
    CELEBRATION = "celebration"


class EventInfo(StrictModel):
    """A dated event or calendar period overlapping the trip."""

    id: str
    name: str
    category: EventCategory
    start_date: date
    end_date: date
    area: str | None = None
    description: str = ""
    approximate: bool = Field(
        default=False, description="Dates vary year to year or depend on moon sighting"
    )
    provenance: Provenance

    @model_validator(mode="after")
    def _range(self) -> EventInfo:
        if self.end_date < self.start_date:
            raise ValueError("event end_date before start_date")
        return self

    def overlaps(self, start: date, end: date) -> bool:
        return self.start_date <= end and start <= self.end_date


class MonthDay(StrictModel):
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)


class RecurringEvent(StrictModel):
    """Seed form of an annual event; ``end`` before ``start`` means it spans New Year."""

    id: str = Field(pattern=r"^evt_[a-z0-9_]+$")
    name: str
    category: EventCategory
    start: MonthDay
    end: MonthDay
    area: str | None = None
    description: str = ""
    odd_years_only: bool = False
    provenance: Provenance


class FixedHoliday(StrictModel):
    id: str = Field(pattern=r"^hol_[a-z0-9_]+$")
    name: str
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    provenance: Provenance


class HijriHoliday(StrictModel):
    """Holiday defined on the Hijri calendar, converted with ``hijridate`` (approximate)."""

    id: str = Field(pattern=r"^hol_[a-z0-9_]+$")
    name: str
    hijri_month: int = Field(ge=1, le=12)
    hijri_day: int = Field(ge=1, le=30)
    duration_days: int = Field(default=1, ge=1, le=10)
    provenance: Provenance


class HolidaySeed(StrictModel):
    fixed: list[FixedHoliday]
    hijri: list[HijriHoliday]
