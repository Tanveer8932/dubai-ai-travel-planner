"""Seed enrichment from OpenStreetMap (used by ``scripts/refresh_seed.py``).

For each curated venue, Nominatim is asked for ``"<name>, <area>"``. A match is accepted
only if it lies within :data:`MAX_SHIFT_KM` of the curated coordinates, so a wrong
same-name hit cannot move a venue across the city. Accepted matches update the
location, ``osm_id``, OSM ``opening_hours`` (when in a simple daily form) and the
``wheelchair`` tag, and record an ``enrichment`` provenance of kind ``api``.
Prices are never touched: OSM has no reliable price data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import Any

from dubai_planner.data.http import HttpClient, SourceUnavailableError
from dubai_planner.data.nominatim import LICENCE, MIN_INTERVAL_S, SEARCH_URL, parse_result
from dubai_planner.models.common import DataKind, GeoPoint, Provenance

MAX_SHIFT_KM = 2.0
_SIMPLE_HOURS = re.compile(r"^(?:Mo-Su\s+)?(\d{2}:\d{2})-(\d{2}:\d{2})$")


class EnrichStatus(StrEnum):
    UPDATED = "updated"
    NO_MATCH = "no_match"
    TOO_FAR = "too_far"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class EnrichOutcome:
    venue_id: str
    status: EnrichStatus
    shift_km: float | None = None


def parse_simple_hours(osm: str) -> tuple[str, str] | None:
    """``"Mo-Su 10:00-22:00"`` or ``"10:00-22:00"`` → ``("10:00", "22:00")``; else None."""
    match = _SIMPLE_HOURS.match(osm.strip())
    if not match:
        return None
    open_, close = match.groups()
    close = "00:00" if close == "24:00" else close
    return open_, close


def enrich_record(
    record: dict[str, Any], area_name: str, http: HttpClient
) -> tuple[dict[str, Any], EnrichOutcome]:
    """Return an updated copy of a raw seed record plus what happened."""
    query = f"{record['name']}, {area_name}"
    params = {"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "ae",
              "extratags": 1, "accept-language": "en"}
    try:
        resp = http.get_json(SEARCH_URL, params)
    except SourceUnavailableError:
        return record, EnrichOutcome(record["id"], EnrichStatus.UNAVAILABLE)
    prov = Provenance(kind=DataKind.API, source="OpenStreetMap Nominatim", source_url=SEARCH_URL,
                      as_of=resp.fetched_at.date(), licence=LICENCE)
    hit = parse_result(query, resp.data, prov)
    if hit is None:
        return record, EnrichOutcome(record["id"], EnrichStatus.NO_MATCH)
    curated = GeoPoint.model_validate(record["location"])
    shift = round(curated.distance_km(hit.location), 3)
    if shift > MAX_SHIFT_KM:
        return record, EnrichOutcome(record["id"], EnrichStatus.TOO_FAR, shift)

    updated = dict(record)
    updated["location"] = {"lat": round(hit.location.lat, 6), "lon": round(hit.location.lon, 6)}
    updated["osm_id"] = hit.osm_id
    fields = ["location", "osm_id"]
    osm_hours = hit.extratags.get("opening_hours")
    if osm_hours:
        simple = parse_simple_hours(osm_hours)
        if simple:
            updated["opening_hours"] = {**(record.get("opening_hours") or {}),
                                        "open": simple[0], "close": simple[1], "osm": osm_hours}
            fields.append("opening_hours")
    wheelchair = hit.extratags.get("wheelchair")
    if wheelchair in {"yes", "no"}:
        updated["wheelchair_accessible"] = wheelchair == "yes"
        fields.append("wheelchair_accessible")
    updated["enrichment"] = Provenance(
        kind=DataKind.API, source="OpenStreetMap Nominatim", source_url=SEARCH_URL,
        as_of=date.today(), licence=LICENCE,
        note=f"Updated {', '.join(fields)}; shift {shift} km from curated point.",
    ).model_dump(mode="json", exclude_none=True)
    return updated, EnrichOutcome(record["id"], EnrichStatus.UPDATED, shift)


def nominatim_http() -> HttpClient:
    """HTTP client honouring the Nominatim policy (1 req/s, UA, cache)."""
    return HttpClient("nominatim", ttl=timedelta(days=30), min_interval_s=MIN_INTERVAL_S)
