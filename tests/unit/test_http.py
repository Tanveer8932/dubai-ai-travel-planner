"""Cached HTTP client: cache hits, TTL, retry, stale fallback, offline mode, rate limiting."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from dubai_planner.config import Settings
from dubai_planner.data.http import (
    HttpClient,
    JsonFileCache,
    RateLimiter,
    SourceUnavailableError,
    limiter_for_host,
)

URL = "https://api.example.test/data"


def _age_cache(client: HttpClient, hours: float) -> None:
    """Make every cached entry of ``client`` look ``hours`` old."""
    for path in (client.cache.root / client.namespace).glob("*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        entry["fetched_at"] = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        path.write_text(json.dumps(entry), encoding="utf-8")


@respx.mock
def test_second_identical_call_is_served_from_cache(make_http: Callable[..., HttpClient]) -> None:
    route = respx.get(URL).respond(json={"v": 1})
    client = make_http()
    first = client.get_json(URL, {"a": 1, "b": 2})
    second = client.get_json(URL, {"b": 2, "a": 1})  # same request, params reordered
    assert first.data == second.data == {"v": 1}
    assert not first.from_cache and second.from_cache
    assert route.call_count == 1 and client.network_calls == 1


@respx.mock
def test_cache_shared_across_instances(make_http: Callable[..., HttpClient]) -> None:
    route = respx.get(URL).respond(json={"v": 1})
    make_http("ns").get_json(URL)
    assert make_http("ns").get_json(URL).from_cache
    assert route.call_count == 1


@respx.mock
def test_user_agent_and_params_sent(
    make_http: Callable[..., HttpClient], settings: Settings
) -> None:
    route = respx.get(URL).respond(json=[])
    make_http().get_json(URL, {"q": "Dubai Frame"})
    request = route.calls.last.request
    assert request.headers["User-Agent"] == settings.http_user_agent
    assert request.url.params["q"] == "Dubai Frame"


@respx.mock
def test_expired_entry_is_refetched(make_http: Callable[..., HttpClient]) -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(200, json={"v": 1}), httpx.Response(200, json={"v": 2})]
    )
    client = make_http(ttl=timedelta(hours=1))
    client.get_json(URL)
    _age_cache(client, hours=2)
    resp = client.get_json(URL)
    assert resp.data == {"v": 2} and not resp.from_cache and route.call_count == 2


@respx.mock
def test_retries_transient_errors(make_http: Callable[..., HttpClient]) -> None:
    route = respx.get(URL).mock(
        side_effect=[
            httpx.ConnectError("boom"),
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    resp = make_http(max_attempts=3).get_json(URL)
    assert resp.data == {"ok": True} and route.call_count == 3


@respx.mock
def test_client_errors_are_not_retried(make_http: Callable[..., HttpClient]) -> None:
    route = respx.get(URL).respond(404)
    with pytest.raises(SourceUnavailableError, match="HTTP 404"):
        make_http().get_json(URL)
    assert route.call_count == 1


@respx.mock
def test_no_content_and_invalid_json(make_http: Callable[..., HttpClient]) -> None:
    respx.get(URL).respond(204)
    with pytest.raises(SourceUnavailableError, match="204"):
        make_http().get_json(URL)
    respx.get(URL + "/bad").respond(200, text="<html>")
    with pytest.raises(SourceUnavailableError, match="invalid JSON"):
        make_http().get_json(URL + "/bad")


@respx.mock
def test_network_failure_uses_stale_cache(make_http: Callable[..., HttpClient]) -> None:
    respx.get(URL).mock(
        side_effect=[
            httpx.Response(200, json={"v": 1}),
            httpx.ConnectError("down"),
            httpx.ConnectError("down"),
            httpx.ConnectError("down"),
        ]
    )
    client = make_http(ttl=timedelta(hours=1))
    client.get_json(URL)
    _age_cache(client, hours=5)
    resp = client.get_json(URL)
    assert resp.data == {"v": 1} and resp.from_cache and resp.stale


@respx.mock
def test_network_failure_without_cache_raises(make_http: Callable[..., HttpClient]) -> None:
    respx.get(URL).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(SourceUnavailableError, match="ConnectError"):
        make_http(max_attempts=2).get_json(URL)


@respx.mock
def test_offline_mode_never_touches_network(make_http: Callable[..., HttpClient]) -> None:
    route = respx.get(URL).respond(json={"v": 1})
    online = make_http("ns")
    online.get_json(URL)
    _age_cache(online, hours=1000)
    offline = make_http("ns", offline=True)
    resp = offline.get_json(URL)
    assert resp.stale and resp.data == {"v": 1}
    with pytest.raises(SourceUnavailableError, match="offline"):
        offline.get_json(URL + "/other")
    assert route.call_count == 1 and offline.network_calls == 0


def test_offline_default_comes_from_settings(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setenv("HTTP_OFFLINE", "true")
    from dubai_planner.config import get_settings

    get_settings.cache_clear()
    assert HttpClient("x").offline is True


def test_corrupt_cache_entry_is_a_miss(tmp_path: object, settings: Settings) -> None:
    cache = JsonFileCache(settings.cache_dir)
    key = cache.make_key("GET", URL, None, None)
    cache.put("ns", key, URL, {"v": 1})
    assert cache.get("ns", key) is not None
    (settings.cache_dir / "ns" / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert cache.get("ns", key) is None


def test_cache_key_depends_on_request() -> None:
    k = JsonFileCache.make_key
    assert k("GET", URL, {"a": 1}, None) == k("get", URL, {"a": 1}, None)
    assert k("GET", URL, {"a": 1}, None) != k("GET", URL, {"a": 2}, None)
    assert k("POST", URL, None, {"data": "x"}) != k("POST", URL, None, {"data": "y"})


def test_rate_limiter_enforces_min_interval() -> None:
    now = [100.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(1.0, clock=lambda: now[0], sleep=sleep)
    limiter.wait()  # first call: no wait
    now[0] += 0.25
    limiter.wait()  # 0.25 s later: must wait the remaining 0.75 s
    now[0] += 2.0
    limiter.wait()  # long gap: no wait
    assert sleeps == [pytest.approx(0.75)]


def test_host_limiters_are_shared() -> None:
    a = limiter_for_host("nominatim.openstreetmap.org", 1.0)
    b = limiter_for_host("nominatim.openstreetmap.org", 1.0)
    assert a is b and a.min_interval_s == 1.0


@respx.mock
def test_cache_hits_do_not_consume_rate_limit(make_http: Callable[..., HttpClient]) -> None:
    respx.get(URL).respond(json={"v": 1})
    client = make_http(min_interval_s=1.0)
    client.get_json(URL)
    limiter = limiter_for_host("api.example.test", 1.0)
    last = limiter._last
    client.get_json(URL)  # cache hit
    assert limiter._last == last


def test_user_agent_without_contact_is_flagged(caplog: pytest.LogCaptureFixture) -> None:
    from dubai_planner.data import http as http_module

    http_module._ua_warned = False
    assert http_module.warn_if_user_agent_lacks_contact("planner/0.1 (academic project)")
    assert "HTTP_USER_AGENT has no contact details" in caplog.text
    assert http_module.warn_if_user_agent_lacks_contact("planner (contact: you@example.com)")
    assert not http_module.warn_if_user_agent_lacks_contact("planner/0.1 (contact: a@b.org)")
    assert not http_module.warn_if_user_agent_lacks_contact("planner/0.1 (+https://x.org)")
