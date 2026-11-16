"""Wikivoyage intro text via the MediaWiki Action API (TextExtracts), CC BY-SA 4.0.

Offline fallback: the curated description of a matching seed area.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any
from urllib.parse import quote

from dubai_planner.data.http import HttpClient, SourceUnavailableError
from dubai_planner.data.seed import load_catalog
from dubai_planner.models.common import DataKind, Provenance
from dubai_planner.models.sources import WikiSummary

logger = logging.getLogger(__name__)

API_URL = "https://en.wikivoyage.org/w/api.php"
PAGE_URL = "https://en.wikivoyage.org/wiki/"
LICENCE = "CC BY-SA 4.0 — Wikivoyage contributors"
MAX_CHARS = 1500


class WikivoyageClient:
    def __init__(self, http: HttpClient | None = None) -> None:
        self.http = http or HttpClient("wikivoyage", ttl=timedelta(days=7))

    def get_summary(self, title: str) -> WikiSummary | None:
        """Intro extract for ``title`` (e.g. ``"Dubai/Deira"``). Never raises for network issues."""
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "redirects": 1,
            "format": "json",
            "formatversion": 2,
            "titles": title,
        }
        try:
            resp = self.http.get_json(API_URL, params)
        except SourceUnavailableError as exc:
            logger.info("wikivoyage unavailable (%s); trying seed areas", exc)
            return seed_summary(title)
        prov = Provenance(
            kind=DataKind.API,
            source="Wikivoyage (MediaWiki API)",
            source_url=PAGE_URL + quote(title.replace(" ", "_")),
            as_of=resp.fetched_at.date(),
            licence=LICENCE,
            note="stale cache" if resp.stale else None,
        )
        return parse_extract(resp.data, prov) or seed_summary(title)


def parse_extract(payload: Any, provenance: Provenance) -> WikiSummary | None:
    try:
        page = payload["query"]["pages"][0]
    except (KeyError, IndexError, TypeError):
        return None
    extract = (page.get("extract") or "").strip()
    if page.get("missing") or not extract:
        return None
    return WikiSummary(
        title=page["title"],
        extract=extract[:MAX_CHARS],
        url=provenance.source_url,
        provenance=provenance,
    )


def seed_summary(title: str) -> WikiSummary | None:
    key = title.split("/")[-1].strip().lower()
    for area in load_catalog().areas:
        if key and (key == area.id.replace("_", " ") or key in area.name.lower()):
            return WikiSummary(title=area.name, extract=area.description, provenance=area.provenance)
    return None
