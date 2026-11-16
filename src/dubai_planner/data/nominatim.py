"""OSM Nominatim geocoding (keyless).

Usage policy (https://operations.osmfoundation.org/policies/nominatim/): at most
1 request per second, a descriptive User-Agent, and cache results. All three are
enforced by :class:`HttpClient` (``min_interval_s=1.0``, ``HTTP_USER_AGENT``, JSON cache).
Offline, names are resolved against seed venues, areas and metro stations.
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Any

from dubai_planner.data.http import HttpClient, SourceUnavailableError
from dubai_planner.data.seed import load_catalog
from dubai_planner.models.common import DataKind, GeoPoint, Provenance
from dubai_planner.models.sources import GeocodeResult

logger = logging.getLogger(__name__)

SEARCH_URL = "https://nominatim.openstreetmap.org/search"
LICENCE = "ODbL 1.0 — © OpenStreetMap contributors"
MIN_INTERVAL_S = 1.0


class NominatimClient:
    def __init__(self, http: HttpClient | None = None) -> None:
        self.http = http or HttpClient(
            "nominatim", ttl=timedelta(days=30), min_interval_s=MIN_INTERVAL_S
        )

    def geocode(self, query: str, *, country_codes: str = "ae") -> GeocodeResult | None:
        """Best match for ``query`` or ``None``. Never raises for network problems."""
        query = query.strip()
        if not query:
            return None
        params = {
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": country_codes,
            "extratags": 1,
            "accept-language": "en",
        }
        try:
            resp = self.http.get_json(SEARCH_URL, params)
        except SourceUnavailableError as exc:
            logger.info("nominatim unavailable (%s); trying seed lookup", exc)
            return seed_lookup(query)
        prov = Provenance(
            kind=DataKind.API,
            source="OpenStreetMap Nominatim",
            source_url=SEARCH_URL,
            as_of=resp.fetched_at.date(),
            licence=LICENCE,
            note="stale cache" if resp.stale else None,
        )
        return parse_result(query, resp.data, prov) or seed_lookup(query)


def parse_result(query: str, payload: Any, provenance: Provenance) -> GeocodeResult | None:
    if not isinstance(payload, list) or not payload:
        return None
    hit = payload[0]
    try:
        location = GeoPoint(lat=float(hit["lat"]), lon=float(hit["lon"]))
    except (KeyError, TypeError, ValueError):
        return None
    extratags = {str(k): str(v) for k, v in (hit.get("extratags") or {}).items()}
    osm_id = f"{hit['osm_type']}/{hit['osm_id']}" if "osm_type" in hit else None
    return GeocodeResult(
        query=query,
        name=hit.get("name") or hit.get("display_name", query),
        location=location,
        display_name=hit.get("display_name", ""),
        osm_id=osm_id,
        category=hit.get("category"),
        extratags=extratags,
        provenance=provenance,
    )


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def seed_lookup(query: str) -> GeocodeResult | None:
    """Match the query against seed venue, area and station names (offline fallback)."""
    q = _norm(query.replace("Dubai", ""))
    if not q:
        return None
    catalog = load_catalog()
    candidates: list[tuple[str, str, GeoPoint, Provenance]] = [
        (v.id, v.name, v.location, v.provenance)
        for v in (*catalog.attractions, *catalog.hotels, *catalog.restaurants)
    ]
    candidates += [(a.id, a.name, a.centroid, a.provenance) for a in catalog.areas]
    candidates += [(s.id, s.name, s.location, s.provenance)
                   for s in catalog.transport.metro_stations]
    for ref_id, name, location, prov in candidates:
        n = _norm(name)
        if q == n or q in n or (len(n) > 4 and n in q):
            return GeocodeResult(
                query=query, name=name, location=location, display_name=f"{name} (seed)",
                osm_id=None, category=ref_id.split("_", 1)[0], provenance=prov,
            )
    return None
