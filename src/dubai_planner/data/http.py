"""Polite, cached HTTP client shared by every public-data source client.

Order of resolution for each request:

1. **Fresh cache hit** (JSON file under ``CACHE_DIR/<namespace>/``, younger than the TTL)
   → returned without touching the network.
2. **Network** (unless ``HTTP_OFFLINE=true``): per-host rate limit (Nominatim ≤ 1 req/s),
   descriptive User-Agent, timeout, and ``tenacity`` retry on transport errors / 429 / 5xx.
3. **Stale cache** (expired entry) if the network fails or is disabled.
4. Otherwise :class:`SourceUnavailableError`; each source client catches it and falls
   back to curated seed data, so callers never see a network exception.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from dubai_planner.config import Settings, get_settings

logger = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
#: Wikimedia and OSM policies require contact details (email or URL) in the User-Agent.
#: Observed 2026-11-15: Wikivoyage answers 403 without them; Overpass answers 406 for the
#: ``you@example.com`` placeholder.
_UA_CONTACT = re.compile(r"@|https?://|contact", re.IGNORECASE)
_ua_warned = False


class SourceUnavailableError(RuntimeError):
    """The source could not be reached and nothing usable was cached."""


class _RetryableError(Exception):
    pass


@dataclass(frozen=True)
class CachedResponse:
    data: Any
    fetched_at: datetime
    from_cache: bool
    stale: bool = False


# --------------------------------------------------------------------------- cache


class JsonFileCache:
    """One JSON file per request, keyed by a SHA-256 of the normalised request."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def make_key(
        method: str, url: str, params: Mapping[str, Any] | None, data: Mapping[str, Any] | None
    ) -> str:
        blob = json.dumps(
            {"m": method.upper(), "u": url, "p": params or {}, "d": data or {}},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{key}.json"

    def get(self, namespace: str, key: str) -> tuple[Any, datetime] | None:
        path = self._path(namespace, key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            return entry["data"], datetime.fromisoformat(entry["fetched_at"])
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError) as exc:  # corrupt entry → treat as a miss
            logger.warning("ignoring unreadable cache entry %s: %s", path.name, exc)
            return None

    def put(self, namespace: str, key: str, url: str, data: Any) -> datetime:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fetched_at = datetime.now(UTC)
        payload = {"url": url, "fetched_at": fetched_at.isoformat(), "data": data}
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic on the same volume
        return fetched_at


# --------------------------------------------------------------------------- rate limit


class RateLimiter:
    """Minimum interval between requests (thread-safe)."""

    def __init__(
        self,
        min_interval_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last is not None:
                remaining = self.min_interval_s - (now - self._last)
                if remaining > 0:
                    self._sleep(remaining)
                    now = self._clock()
            self._last = now


_host_limiters: dict[str, RateLimiter] = {}
_host_lock = threading.Lock()


def limiter_for_host(host: str, min_interval_s: float) -> RateLimiter:
    """Process-wide limiter per host, so separate client instances share the budget."""
    with _host_lock:
        limiter = _host_limiters.get(host)
        if limiter is None or limiter.min_interval_s < min_interval_s:
            limiter = RateLimiter(min_interval_s)
            _host_limiters[host] = limiter
        return limiter


# --------------------------------------------------------------------------- client


def warn_if_user_agent_lacks_contact(user_agent: str) -> bool:
    """Log once if ``HTTP_USER_AGENT`` has no contact info; returns True if it lacks one."""
    global _ua_warned
    placeholder = "example.com" in user_agent.lower()
    if _UA_CONTACT.search(user_agent) and not placeholder:
        return False
    if not _ua_warned:
        problem = "uses the example.com placeholder" if placeholder else "has no contact details"
        logger.warning(
            "HTTP_USER_AGENT %s. Wikivoyage/Wikipedia reject requests without contact info "
            "(HTTP 403) and Overpass rejects the placeholder (HTTP 406); clients will fall back "
            "to seed data. Set a real contact in .env, e.g. "
            "'dubai-ai-travel-planner/0.1 (academic project; contact: you@your-university.edu)'.",
            problem,
        )
        _ua_warned = True
    return True


class HttpClient:
    """Cached JSON HTTP client for one data source (``namespace`` = cache sub-folder)."""

    def __init__(
        self,
        namespace: str,
        *,
        settings: Settings | None = None,
        ttl: timedelta | None = None,
        min_interval_s: float = 0.0,
        max_attempts: int = 3,
        backoff_s: float = 0.5,
        offline: bool | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or get_settings()
        self.namespace = namespace
        self.cache = JsonFileCache(self.settings.cache_dir)
        self.ttl = ttl if ttl is not None else timedelta(hours=self.settings.cache_ttl_hours)
        self.min_interval_s = min_interval_s
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.offline = self.settings.http_offline if offline is None else offline
        self._sleep = sleep
        self._client: httpx.Client | None = None
        self.network_calls = 0  # observable in tests
        warn_if_user_agent_lacks_contact(self.settings.http_user_agent)

    # -- lifecycle
    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={"User-Agent": self.settings.http_user_agent, "Accept": "application/json"},
                timeout=self.settings.http_timeout_s,
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- public API
    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> CachedResponse:
        return self._request("GET", url, params=params, data=None, headers=headers)

    def post_form_json(
        self,
        url: str,
        data: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> CachedResponse:
        return self._request("POST", url, params=None, data=data, headers=headers)

    # -- internals
    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None,
        data: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
    ) -> CachedResponse:
        key = self.cache.make_key(method, url, params, data)
        cached = self.cache.get(self.namespace, key)
        if cached is not None:
            body, fetched_at = cached
            if datetime.now(UTC) - fetched_at < self.ttl:
                return CachedResponse(body, fetched_at, from_cache=True)

        if self.offline:
            return self._stale_or_raise(cached, url, "offline mode (HTTP_OFFLINE=true)")

        try:
            body = self._fetch_with_retry(method, url, params, data, headers)
        except (SourceUnavailableError, _RetryableError, httpx.HTTPError) as exc:
            return self._stale_or_raise(cached, url, f"{type(exc).__name__}: {exc}")

        fetched_at = self.cache.put(self.namespace, key, url, body)
        return CachedResponse(body, fetched_at, from_cache=False)

    def _stale_or_raise(
        self, cached: tuple[Any, datetime] | None, url: str, reason: str
    ) -> CachedResponse:
        if cached is not None:
            logger.info("%s: using stale cache for %s (%s)", self.namespace, url, reason)
            return CachedResponse(cached[0], cached[1], from_cache=True, stale=True)
        raise SourceUnavailableError(f"{self.namespace}: {url} unavailable ({reason})")

    def _fetch_with_retry(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None,
        data: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
    ) -> Any:
        limiter = (
            limiter_for_host(urlsplit(url).netloc, self.min_interval_s)
            if self.min_interval_s > 0
            else None
        )
        retrying = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=self.backoff_s, max=10),
            retry=retry_if_exception_type((_RetryableError, httpx.TransportError)),
            sleep=self._sleep,
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                if limiter is not None:
                    limiter.wait()
                self.network_calls += 1
                response = self._http().request(
                    method, url, params=params, data=data, headers=headers
                )
                if response.status_code in RETRY_STATUS:
                    raise _RetryableError(f"HTTP {response.status_code}")
                if response.status_code == 204:
                    raise SourceUnavailableError(f"{url}: no content (HTTP 204)")
                if response.is_error:
                    raise SourceUnavailableError(f"{url}: HTTP {response.status_code}")
                try:
                    return response.json()
                except ValueError as exc:
                    raise SourceUnavailableError(f"{url}: invalid JSON") from exc
        raise SourceUnavailableError(f"{url}: retries exhausted")  # pragma: no cover
