"""OpenStreetMap Overpass API (keyless, fair use): points of interest around Dubai.

Data © OpenStreetMap contributors, ODbL 1.0. On network failure, POIs are derived
from curated seed records so callers always get a usable (if smaller) list.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from enum import StrEnum
from typing import Any

from dubai_planner.data.http import HttpClient, SourceUnavailableError
from dubai_planner.data.seed import load_catalog
from dubai_planner.models.common import DUBAI_BBOX, DataKind, GeoPoint, Provenance
from dubai_planner.models.sources import OsmPoi

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
LICENCE = "ODbL 1.0 — © OpenStreetMap contributors"


class PoiCategory(StrEnum):
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    METRO_STATION = "metro_station"


#: Overpass tag filters per category (each entry becomes one nwr[...] clause).
_FILTERS: dict[PoiCategory, list[str]] = {
    PoiCategory.ATTRACTION: [
        '["tourism"~"^(attraction|museum|theme_park|zoo|aquarium|viewpoint|gallery)$"]'
    ],
    PoiCategory.RESTAURANT: ['["amenity"="restaurant"]'],
    PoiCategory.METRO_STATION: ['["railway"="station"]["station"="subway"]'],
}


def build_query(
    category: PoiCategory,
    bbox: tuple[float, float, float, float] = DUBAI_BBOX,
    limit: int = 200,
) -> str:
    south, west, north, east = bbox
    box = f"({south},{west},{north},{east})"
    clauses = "".join(f"nwr{f}[name]{box};" for f in _FILTERS[category])
    return f"[out:json][timeout:25];({clauses});out center tags {limit};"


class OverpassClient:
    def __init__(self, http: HttpClient | None = None) -> None:
        # OSM data changes slowly; cache for a week and space requests politely.
        self.http = http or HttpClient("overpass", ttl=timedelta(days=7), min_interval_s=1.0)

    def search_pois(
        self,
        category: PoiCategory,
        bbox: tuple[float, float, float, float] = DUBAI_BBOX,
        limit: int = 200,
    ) -> list[OsmPoi]:
        """Never raises for network problems; falls back to seed-derived POIs."""
        try:
            resp = self.http.post_form_json(OVERPASS_URL, {"data": build_query(category, bbox, limit)})
            prov = Provenance(
                kind=DataKind.API,
                source="OpenStreetMap Overpass API",
                source_url=OVERPASS_URL,
                as_of=resp.fetched_at.date(),
                licence=LICENCE,
                note="stale cache" if resp.stale else None,
            )
            return parse_elements(resp.data, prov)
        except (SourceUnavailableError, KeyError, TypeError) as exc:
            logger.info("overpass unavailable (%s); using seed POIs", exc)
            return seed_pois(category)


def parse_elements(payload: Any, provenance: Provenance) -> list[OsmPoi]:
    pois: list[OsmPoi] = []
    for el in payload["elements"]:
        tags = {str(k): str(v) for k, v in (el.get("tags") or {}).items()}
        name = tags.get("name:en") or tags.get("name")
        lat = el.get("lat", (el.get("center") or {}).get("lat"))
        lon = el.get("lon", (el.get("center") or {}).get("lon"))
        if not name or lat is None or lon is None:
            continue
        pois.append(
            OsmPoi(
                osm_id=f"{el['type']}/{el['id']}",
                name=name,
                location=GeoPoint(lat=lat, lon=lon),
                tags=tags,
                provenance=provenance,
            )
        )
    return pois


def seed_pois(category: PoiCategory) -> list[OsmPoi]:
    catalog = load_catalog()
    if category is PoiCategory.ATTRACTION:
        return [
            OsmPoi(osm_id=a.osm_id or f"seed/{a.id}", name=a.name, location=a.location,
                   tags={"tourism": a.category.value}, ref_id=a.id, provenance=a.provenance)
            for a in catalog.attractions
        ]
    if category is PoiCategory.RESTAURANT:
        return [
            OsmPoi(osm_id=r.osm_id or f"seed/{r.id}", name=r.name, location=r.location,
                   tags={"amenity": "restaurant", "cuisine": ";".join(r.cuisines)},
                   ref_id=r.id, provenance=r.provenance)
            for r in catalog.restaurants
        ]
    return [
        OsmPoi(osm_id=f"seed/{s.id}", name=s.name, location=s.location,
               tags={"railway": "station", "lines": ";".join(s.lines)}, ref_id=s.id,
               provenance=s.provenance)
        for s in catalog.transport.metro_stations
    ]
