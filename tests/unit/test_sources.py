"""AC1.3 / AC1.6: each source client parses mocked HTTP, caches, and falls back to seed."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal

import httpx
import pytest
import respx

from dubai_planner.data import calendar, fx, nominatim, open_meteo, overpass, wikivoyage
from dubai_planner.data.fx import FxClient
from dubai_planner.data.http import HttpClient
from dubai_planner.data.nominatim import NominatimClient
from dubai_planner.data.open_meteo import OpenMeteoClient
from dubai_planner.data.overpass import OverpassClient, PoiCategory, build_query
from dubai_planner.data.wikivoyage import WikivoyageClient
from dubai_planner.models import DataKind, EventCategory, WeatherSource

TODAY = date(2026, 11, 15)

# ----------------------------------------------------------------------------- Open-Meteo


def _daily(dates: list[str], tmax: list[float], uv: bool = True) -> dict[str, object]:
    daily: dict[str, object] = {
        "time": dates,
        "temperature_2m_max": tmax,
        "temperature_2m_min": [t - 8 for t in tmax],
        "precipitation_sum": [0.0] * len(dates),
    }
    if uv:
        daily["uv_index_max"] = [6.0] * len(dates)
    return {"latitude": 25.2, "longitude": 55.3, "timezone": "Asia/Dubai", "daily": daily}


@respx.mock
def test_open_meteo_forecast_and_cache(make_http: Callable[..., HttpClient]) -> None:
    route = respx.get(open_meteo.FORECAST_URL).respond(
        json=_daily(["2026-11-16", "2026-11-17"], [28.8, 27.7])
    )
    client = OpenMeteoClient(make_http("open_meteo"), today=TODAY)
    ctx = client.get_weather_context(date(2026, 11, 16), date(2026, 11, 17))
    assert ctx.source is WeatherSource.FORECAST
    assert [d.temp_max_c for d in ctx.days] == [28.8, 27.7]
    assert ctx.provenance.kind is DataKind.API and "Open-Meteo" in ctx.provenance.source
    params = route.calls.last.request.url.params
    assert params["start_date"] == "2026-11-16" and params["timezone"] == "Asia/Dubai"
    # cache hit: identical call makes no request
    client.get_weather_context(date(2026, 11, 16), date(2026, 11, 17))
    assert route.call_count == 1


@respx.mock
def test_open_meteo_far_future_uses_last_years_archive(
    make_http: Callable[..., HttpClient],
) -> None:
    route = respx.get(open_meteo.ARCHIVE_URL).respond(
        json=_daily(["2026-07-01", "2026-07-02"], [41.0, 42.4], uv=False)
    )
    client = OpenMeteoClient(make_http("open_meteo"), today=TODAY)
    ctx = client.get_weather_context(date(2027, 7, 1), date(2027, 7, 2))
    assert route.calls.last.request.url.params["start_date"] == "2026-07-01"
    assert ctx.source is WeatherSource.HISTORICAL
    assert [d.date for d in ctx.days] == [date(2027, 7, 1), date(2027, 7, 2)]
    assert ctx.provenance.kind is DataKind.ESTIMATED  # proxy, not a forecast
    assert ctx.hot_days == [date(2027, 7, 1), date(2027, 7, 2)]
    client.get_weather_context(date(2027, 7, 1), date(2027, 7, 2))
    assert route.call_count == 1


@respx.mock
def test_open_meteo_offline_falls_back_to_climatology(
    make_http: Callable[..., HttpClient],
) -> None:
    respx.get(open_meteo.FORECAST_URL).mock(side_effect=httpx.ConnectError("offline"))
    client = OpenMeteoClient(make_http("open_meteo"), today=TODAY)
    ctx = client.get_weather_context(date(2026, 11, 20), date(2026, 11, 22))
    assert ctx.source is WeatherSource.CLIMATOLOGY and len(ctx.days) == 3
    assert ctx.provenance.kind is DataKind.ESTIMATED


@respx.mock
def test_open_meteo_malformed_payload_falls_back(make_http: Callable[..., HttpClient]) -> None:
    respx.get(open_meteo.FORECAST_URL).respond(json={"error": True, "reason": "bad"})
    ctx = OpenMeteoClient(make_http("open_meteo"), today=TODAY).get_weather_context(
        date(2026, 11, 16), date(2026, 11, 16)
    )
    assert ctx.source is WeatherSource.CLIMATOLOGY


def test_climatology_july_is_hot() -> None:
    ctx = open_meteo.climatology(date(2027, 7, 1), date(2027, 7, 3))
    assert all(d.is_hot for d in ctx.days)


# ----------------------------------------------------------------------------- Overpass

OVERPASS_PAYLOAD = {
    "elements": [
        {
            "type": "node",
            "id": 3717446089,
            "lat": 25.2635,
            "lon": 55.2972,
            "tags": {
                "name": "متحف دبي",
                "name:en": "Dubai Museum",
                "tourism": "museum",
                "wheelchair": "yes",
            },
        },
        {
            "type": "way",
            "id": 386649647,
            "center": {"lat": 25.2355, "lon": 55.3004},
            "tags": {"name": "Dubai Frame", "tourism": "attraction"},
        },
        {"type": "node", "id": 1, "lat": 25.0, "lon": 55.0, "tags": {"tourism": "museum"}},
    ]
}


def test_overpass_query_shape() -> None:
    q = build_query(PoiCategory.RESTAURANT, (1.0, 2.0, 3.0, 4.0), limit=5)
    assert q.startswith("[out:json]") and '["amenity"="restaurant"]' in q
    assert "(1.0,2.0,3.0,4.0)" in q and q.endswith("out center tags 5;")


@respx.mock
def test_overpass_parses_nodes_and_ways_and_caches(make_http: Callable[..., HttpClient]) -> None:
    route = respx.post(overpass.OVERPASS_URL).respond(json=OVERPASS_PAYLOAD)
    client = OverpassClient(make_http("overpass"))
    pois = client.search_pois(PoiCategory.ATTRACTION)
    assert [p.name for p in pois] == ["Dubai Museum", "Dubai Frame"]  # unnamed node skipped
    assert pois[1].osm_id == "way/386649647" and pois[1].location.lat == 25.2355
    assert pois[0].provenance.kind is DataKind.API
    assert pois[0].provenance.licence and "ODbL" in pois[0].provenance.licence
    assert "data=" in route.calls.last.request.content.decode()
    client.search_pois(PoiCategory.ATTRACTION)
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize("category", list(PoiCategory))
def test_overpass_offline_falls_back_to_seed(
    make_http: Callable[..., HttpClient], category: PoiCategory
) -> None:
    respx.post(overpass.OVERPASS_URL).mock(side_effect=httpx.ConnectError("offline"))
    pois = OverpassClient(make_http("overpass")).search_pois(category)
    assert len(pois) >= 20 and all(p.ref_id for p in pois)
    assert all(p.provenance.kind is DataKind.CURATED for p in pois)


# ----------------------------------------------------------------------------- Nominatim

NOMINATIM_PAYLOAD = [
    {
        "osm_type": "way",
        "osm_id": 386649647,
        "lat": "25.2354560",
        "lon": "55.3003671",
        "category": "tourism",
        "type": "museum",
        "name": "Dubai Frame",
        "display_name": "Dubai Frame, Zabeel, Dubai, United Arab Emirates",
        "extratags": {"fee": "yes", "opening_hours": "Mo-Su 09:00-21:00", "wheelchair": "yes"},
    }
]


@respx.mock
def test_nominatim_geocode_parses_and_caches(make_http: Callable[..., HttpClient]) -> None:
    route = respx.get(nominatim.SEARCH_URL).respond(json=NOMINATIM_PAYLOAD)
    client = NominatimClient(make_http("nominatim", min_interval_s=1.0))
    hit = client.geocode("Dubai Frame")
    assert hit is not None and hit.osm_id == "way/386649647"
    assert hit.location.lat == pytest.approx(25.235456)
    assert hit.extratags["opening_hours"] == "Mo-Su 09:00-21:00"
    params = route.calls.last.request.url.params
    assert params["countrycodes"] == "ae" and params["format"] == "jsonv2"
    assert client.geocode("Dubai Frame") == hit
    assert route.call_count == 1


def test_nominatim_client_respects_policy() -> None:
    client = NominatimClient()
    assert client.http.min_interval_s >= 1.0  # ≤ 1 request per second
    assert client.http.settings.http_user_agent  # descriptive UA always sent


@respx.mock
def test_nominatim_offline_uses_seed_lookup(make_http: Callable[..., HttpClient]) -> None:
    respx.get(nominatim.SEARCH_URL).mock(side_effect=httpx.ConnectError("offline"))
    client = NominatimClient(make_http("nominatim"))
    hit = client.geocode("Dubai Frame, Dubai")
    assert hit is not None and hit.name == "Dubai Frame"
    assert hit.provenance.kind is DataKind.CURATED
    assert client.geocode("Union") is not None  # metro station
    assert client.geocode("Totally Unknown Place 123") is None
    assert client.geocode("   ") is None


@respx.mock
def test_nominatim_empty_result_falls_back_to_seed(make_http: Callable[..., HttpClient]) -> None:
    respx.get(nominatim.SEARCH_URL).respond(json=[])
    hit = NominatimClient(make_http("nominatim")).geocode("Global Village")
    assert hit is not None and hit.display_name.endswith("(seed)")


# ----------------------------------------------------------------------------- Wikivoyage


@respx.mock
def test_wikivoyage_summary_and_cache(make_http: Callable[..., HttpClient]) -> None:
    route = respx.get(wikivoyage.API_URL).respond(
        json={
            "query": {
                "pages": [
                    {"pageid": 10092, "title": "Dubai/Deira", "extract": "Deira is the old centre."}
                ]
            }
        }
    )
    client = WikivoyageClient(make_http("wikivoyage"))
    s = client.get_summary("Dubai/Deira")
    assert s is not None and s.extract == "Deira is the old centre."
    assert s.provenance.licence and "CC BY-SA" in s.provenance.licence
    assert s.url == "https://en.wikivoyage.org/wiki/Dubai/Deira"
    client.get_summary("Dubai/Deira")
    assert route.call_count == 1


@respx.mock
def test_wikivoyage_missing_page_and_offline(make_http: Callable[..., HttpClient]) -> None:
    respx.get(wikivoyage.API_URL).mock(
        side_effect=[
            httpx.Response(200, json={"query": {"pages": [{"title": "Nowhere", "missing": True}]}}),
            httpx.ConnectError("offline"),
            httpx.ConnectError("offline"),
            httpx.ConnectError("x"),
        ]
    )
    client = WikivoyageClient(make_http("wikivoyage"))
    assert client.get_summary("Nowhere") is None
    fallback = client.get_summary("Dubai/Deira")
    assert fallback is not None and fallback.title == "Deira"
    assert fallback.provenance.kind is DataKind.CURATED


# ----------------------------------------------------------------------------- FX


def test_ac16_usd_to_aed_uses_peg_without_network() -> None:
    with respx.mock(assert_all_called=False) as mock:
        assert fx.convert(100, "USD", "AED") == 367.25
        assert fx.convert(100, "USD", "AED") == Decimal("367.25")
        assert fx.convert("367.25", "aed", "usd") == Decimal("100.00")
        assert fx.convert(5, "AED", "AED") == Decimal("5.00")
        assert not mock.calls


@respx.mock
def test_fx_cross_rate_via_usd_and_cache(make_http: Callable[..., HttpClient]) -> None:
    route = respx.get(fx.LATEST_URL).respond(
        json={
            "amount": 1.0,
            "base": "USD",
            "date": "2026-11-13",
            "rates": {"EUR": 0.90481, "GBP": 0.75605},
        }
    )
    client = FxClient(make_http("frankfurter"))
    # 100 EUR → 100 / 0.90481 USD → × 3.6725 AED = 405.886...
    assert client.convert(100, "EUR", "AED") == Decimal("405.89")
    rates = client.rates()
    assert rates.rates["AED"] == Decimal("3.6725") and rates.provenance.kind is DataKind.API
    assert rates.provenance.as_of == date(2026, 11, 13)
    assert route.call_count == 1  # second lookup served from cache


@respx.mock
def test_fx_offline_uses_seed_snapshot(make_http: Callable[..., HttpClient]) -> None:
    respx.get(fx.LATEST_URL).mock(side_effect=httpx.ConnectError("offline"))
    client = FxClient(make_http("frankfurter"))
    rates = client.rates()
    assert "EUR" in rates.rates and rates.rates["AED"] == Decimal("3.6725")
    assert rates.provenance.note and "snapshot" in rates.provenance.note
    assert client.convert(100, "GBP", "AED") > Decimal("400")


def test_fx_rejects_bad_input() -> None:
    offline_client = FxClient(HttpClient("frankfurter", offline=True))
    with pytest.raises(ValueError, match="unsupported currency 'XXX'"):
        fx.convert(1, "XXX", "AED", client=offline_client)
    with pytest.raises(ValueError, match="invalid amount"):
        fx.convert("abc", "USD", "AED")


def test_aed_peg_is_verified_provenance() -> None:
    assert Decimal("3.6725") == fx.AED_PER_USD
    assert fx.PEG_PROVENANCE.kind is DataKind.VERIFIED and fx.PEG_PROVENANCE.source_url


# ----------------------------------------------------------------------------- calendar


def test_ramadan_window_1447() -> None:
    start, end = calendar.ramadan_window(1447)
    assert start == date(2026, 2, 18)
    assert 29 <= (end - start).days + 1 <= 30


def test_ramadan_overlap_flagged_approximate() -> None:
    ev = calendar.ramadan_overlap(date(2026, 3, 1), date(2026, 3, 5))
    assert ev is not None and ev.category is EventCategory.RELIGIOUS and ev.approximate
    assert ev.provenance.kind is DataKind.ESTIMATED
    assert calendar.ramadan_overlap(date(2026, 11, 1), date(2026, 11, 5)) is None


def test_public_holidays_fixed_and_hijri() -> None:
    hols = calendar.public_holidays(date(2026, 11, 28), date(2026, 12, 5))
    assert {h.name for h in hols} >= {"Commemoration Day", "UAE National Day"}
    eid = calendar.public_holidays(date(2026, 3, 15), date(2026, 3, 25))
    assert any(h.name == "Eid al-Fitr" and h.approximate for h in eid)


def test_recurring_events_span_new_year_and_odd_years() -> None:
    jan = calendar.get_events(date(2027, 1, 3), date(2027, 1, 6))
    dsf = [e for e in jan if e.name == "Dubai Shopping Festival"]
    assert dsf and dsf[0].start_date == date(2026, 12, 6) and dsf[0].end_date.year == 2027
    assert any(e.name == "Global Village season" for e in jan)
    nov_odd = calendar.get_events(date(2027, 11, 15), date(2027, 11, 16))
    nov_even = calendar.get_events(date(2026, 11, 15), date(2026, 11, 16))
    assert any(e.name == "Dubai Airshow" for e in nov_odd)
    assert not any(e.name == "Dubai Airshow" for e in nov_even)


def test_get_events_validates_range() -> None:
    with pytest.raises(ValueError):
        calendar.get_events(date(2027, 1, 2), date(2027, 1, 1))
    assert calendar.get_events(date(2027, 1, 1), date(2027, 1, 1) + timedelta(days=0))
