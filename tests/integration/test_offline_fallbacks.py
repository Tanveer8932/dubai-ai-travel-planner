"""AC1.4: with the network disabled, every client falls back to seed/cache without raising."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import httpx
import pytest
import respx

from dubai_planner.config import get_settings
from dubai_planner.data import calendar, fx
from dubai_planner.data.fx import FxClient
from dubai_planner.data.http import HttpClient
from dubai_planner.data.nominatim import NominatimClient
from dubai_planner.data.open_meteo import OpenMeteoClient
from dubai_planner.data.overpass import OverpassClient, PoiCategory
from dubai_planner.data.wikivoyage import WikivoyageClient
from dubai_planner.models import WeatherSource

START, END = date(2027, 7, 1), date(2027, 7, 3)


def _exercise_all(http: Callable[[str], HttpClient]) -> None:
    weather = OpenMeteoClient(http("open_meteo"), today=date(2026, 11, 15)).get_weather_context(
        START, END
    )
    assert weather.source is WeatherSource.CLIMATOLOGY and len(weather.days) == 3
    assert OverpassClient(http("overpass")).search_pois(PoiCategory.RESTAURANT)
    hit = NominatimClient(http("nominatim")).geocode("Burj Khalifa")
    assert hit is not None
    assert WikivoyageClient(http("wikivoyage")).get_summary("Dubai/Deira") is not None
    fx_client = FxClient(http("frankfurter"))
    assert fx_client.rates().rates["EUR"] > 0
    assert fx.convert(100, "USD", "AED") == 367.25
    assert calendar.get_events(date(2027, 2, 1), date(2027, 3, 31))  # offline by design


def test_offline_mode_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP_OFFLINE=true: no request is attempted at all."""
    monkeypatch.setenv("HTTP_OFFLINE", "true")
    get_settings.cache_clear()
    with respx.mock(assert_all_called=False) as mock:
        mock.route().mock(side_effect=AssertionError("network must not be used"))
        _exercise_all(lambda ns: HttpClient(ns))
        assert not mock.calls


def test_connection_errors_everywhere(make_http: Callable[..., HttpClient]) -> None:
    """Network 'up' in config but every connection fails (cable unplugged)."""
    with respx.mock(assert_all_called=False) as mock:
        mock.route().mock(side_effect=httpx.ConnectError("network unreachable"))
        _exercise_all(lambda ns: make_http(ns, max_attempts=2))
        assert mock.calls  # it really tried, then fell back


def test_default_clients_survive_blocked_sockets() -> None:
    """Default-constructed clients (real httpx transport, sockets blocked) don't raise."""
    client = OpenMeteoClient(HttpClient("open_meteo", sleep=lambda _s: None))
    weather = client.get_weather_context(START, END)
    assert weather.source is WeatherSource.CLIMATOLOGY
    assert (
        NominatimClient(HttpClient("nominatim", sleep=lambda _s: None)).geocode("Dubai Mall")
        is not None
    )
