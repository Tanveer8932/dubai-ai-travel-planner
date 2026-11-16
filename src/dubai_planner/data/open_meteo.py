"""Open-Meteo weather (keyless): forecast → same dates last year (archive) → seed climatology.

- Trip within the forecast horizon (≤ 16 days ahead): Forecast API.
- Otherwise: Historical (ERA5) archive for the same calendar dates one year earlier,
  as a realistic proxy for typical conditions.
- Network unavailable and nothing cached: monthly climate normals from ``climate.json``.

Open-Meteo data is CC BY 4.0; attribution is kept in the provenance record.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from dubai_planner.data.http import HttpClient, SourceUnavailableError
from dubai_planner.data.seed import load_catalog
from dubai_planner.models.common import DUBAI_CENTRE, DataKind, GeoPoint, Provenance
from dubai_planner.models.context import DailyWeather, WeatherContext, WeatherSource

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HORIZON_DAYS = 16
LICENCE = "CC BY 4.0 (Open-Meteo)"


def _shift_year(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # 29 Feb → 28 Feb
        return day.replace(year=day.year + years, day=28)


class OpenMeteoClient:
    def __init__(self, http: HttpClient | None = None, *, today: date | None = None) -> None:
        self.http = http or HttpClient("open_meteo", ttl=timedelta(hours=6))
        self._today = today

    @property
    def today(self) -> date:
        return self._today or date.today()

    def get_weather_context(
        self, start: date, end: date, location: GeoPoint = DUBAI_CENTRE
    ) -> WeatherContext:
        """Never raises for network problems; falls back to climatology."""
        if end < start:
            raise ValueError("end must not be before start")
        try:
            if (start - self.today).days >= 0 and (end - self.today).days < FORECAST_HORIZON_DAYS:
                return self._forecast(start, end, location)
            if start <= self.today and end < self.today:
                return self._archive(start, end, location, shift_years=0)
            return self._archive(start, end, location, shift_years=-1)
        except (SourceUnavailableError, KeyError, ValueError) as exc:
            logger.info("open-meteo unavailable (%s); using seed climatology", exc)
            return climatology(start, end)

    # -- network paths
    def _params(self, start: date, end: date, location: GeoPoint, daily: str) -> dict[str, Any]:
        return {
            "latitude": location.lat,
            "longitude": location.lon,
            "daily": daily,
            "timezone": "Asia/Dubai",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }

    def _forecast(self, start: date, end: date, location: GeoPoint) -> WeatherContext:
        daily = "temperature_2m_max,temperature_2m_min,precipitation_sum,uv_index_max"
        resp = self.http.get_json(FORECAST_URL, self._params(start, end, location, daily))
        days = _parse_daily(resp.data, shift_years=0)
        return WeatherContext(
            source=WeatherSource.FORECAST,
            days=days,
            provenance=Provenance(
                kind=DataKind.API,
                source="Open-Meteo Forecast API",
                source_url=FORECAST_URL,
                as_of=resp.fetched_at.date(),
                licence=LICENCE,
                note="stale cache" if resp.stale else None,
            ),
        )

    def _archive(
        self, start: date, end: date, location: GeoPoint, *, shift_years: int
    ) -> WeatherContext:
        q_start, q_end = _shift_year(start, shift_years), _shift_year(end, shift_years)
        daily = "temperature_2m_max,temperature_2m_min,precipitation_sum"
        resp = self.http.get_json(ARCHIVE_URL, self._params(q_start, q_end, location, daily))
        days = _parse_daily(resp.data, shift_years=-shift_years)
        note = (
            f"Observed weather for the same dates in {q_start.year}, used as an estimate"
            if shift_years
            else "Observed (historical) weather"
        )
        return WeatherContext(
            source=WeatherSource.HISTORICAL,
            days=[d for d in days if start <= d.date <= end],
            provenance=Provenance(
                kind=DataKind.API if shift_years == 0 else DataKind.ESTIMATED,
                source="Open-Meteo Historical Weather API (ERA5)",
                source_url=ARCHIVE_URL,
                as_of=resp.fetched_at.date(),
                licence=LICENCE,
                note=note,
            ),
        )


def _parse_daily(payload: Any, *, shift_years: int) -> list[DailyWeather]:
    daily = payload["daily"]
    uv = daily.get("uv_index_max") or [None] * len(daily["time"])
    days: list[DailyWeather] = []
    for i, iso in enumerate(daily["time"]):
        tmax, tmin = daily["temperature_2m_max"][i], daily["temperature_2m_min"][i]
        if tmax is None or tmin is None:
            raise ValueError(f"missing temperature for {iso}")
        days.append(
            DailyWeather(
                date=_shift_year(date.fromisoformat(iso), shift_years),
                temp_max_c=tmax,
                temp_min_c=tmin,
                precipitation_mm=daily["precipitation_sum"][i] or 0.0,
                uv_index_max=uv[i],
            )
        )
    if not days:
        raise ValueError("no daily data returned")
    return days


def climatology(start: date, end: date) -> WeatherContext:
    """Offline fallback: monthly normals from seed for each trip day."""
    normals = load_catalog().climate
    days = []
    day = start
    while day <= end:
        m = normals.for_month(day.month)
        days.append(
            DailyWeather(
                date=day,
                temp_max_c=m.avg_high_c,
                temp_min_c=m.avg_low_c,
                precipitation_mm=round(m.precipitation_mm / 30, 2),
            )
        )
        day += timedelta(days=1)
    return WeatherContext(
        source=WeatherSource.CLIMATOLOGY,
        days=days,
        provenance=normals.provenance.model_copy(
            update={"kind": DataKind.ESTIMATED, "note": "Monthly climate normals (offline)"}
        ),
    )


def get_weather_context(start: date, end: date) -> WeatherContext:
    """Convenience wrapper used by tools."""
    return OpenMeteoClient().get_weather_context(start, end)
