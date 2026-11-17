"""Seed enrichment from OSM (refresh_seed.py logic) and the init_db script."""

from __future__ import annotations

import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import respx

from dubai_planner.config import PROJECT_ROOT
from dubai_planner.data import nominatim
from dubai_planner.data.enrichment import EnrichStatus, enrich_record, parse_simple_hours
from dubai_planner.data.http import HttpClient
from dubai_planner.models import AttractionOption, DataKind
from dubai_planner.storage.repository import SCHEMA_VERSION


def _frame_record() -> dict[str, Any]:
    doc = json.loads((PROJECT_ROOT / "data/seed/attractions.json").read_text(encoding="utf-8"))
    rec: dict[str, Any] = next(r for r in doc["items"] if r["id"] == "att_dubai_frame")
    return rec


def _hit(lat: str, lon: str, **extratags: str) -> list[dict[str, Any]]:
    return [
        {
            "osm_type": "way",
            "osm_id": 386649647,
            "lat": lat,
            "lon": lon,
            "name": "Dubai Frame",
            "display_name": "Dubai Frame",
            "extratags": extratags,
        }
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Mo-Su 09:00-21:00", ("09:00", "21:00")),
        ("10:00-24:00", ("10:00", "00:00")),
        ("Mo-Fr 09:00-17:00; Sa 10:00-14:00", None),
        ("24/7", None),
    ],
)
def test_parse_simple_hours(raw: str, expected: tuple[str, str] | None) -> None:
    assert parse_simple_hours(raw) == expected


@respx.mock
def test_enrich_updates_nearby_match(make_http: Callable[..., HttpClient]) -> None:
    respx.get(nominatim.SEARCH_URL).respond(
        json=_hit("25.2354560", "55.3003671", opening_hours="Mo-Su 08:00-22:00", wheelchair="yes")
    )
    updated, outcome = enrich_record(_frame_record(), "Karama & Zabeel", make_http("nominatim"))
    assert outcome.status is EnrichStatus.UPDATED and outcome.shift_km is not None
    venue = AttractionOption.model_validate(updated)  # still a valid seed record
    assert venue.osm_id == "way/386649647"
    assert venue.opening_hours is not None and venue.opening_hours.osm == "Mo-Su 08:00-22:00"
    assert venue.wheelchair_accessible is True
    assert venue.enrichment is not None and venue.enrichment.kind is DataKind.API
    assert venue.price == AttractionOption.model_validate(_frame_record()).price  # untouched


@respx.mock
def test_enrich_rejects_far_or_missing_match(make_http: Callable[..., HttpClient]) -> None:
    respx.get(nominatim.SEARCH_URL).respond(json=_hit("24.4539", "54.3773"))  # Abu Dhabi
    rec = _frame_record()
    same, outcome = enrich_record(rec, "Karama", make_http("nominatim"))
    assert outcome.status is EnrichStatus.TOO_FAR and same is rec
    respx.get(nominatim.SEARCH_URL).respond(json=[])
    _, outcome = enrich_record(rec, "Elsewhere", make_http("nominatim2"))
    assert outcome.status is EnrichStatus.NO_MATCH


def test_enrich_offline_is_unavailable() -> None:
    _, outcome = enrich_record(_frame_record(), "Karama", HttpClient("nominatim", offline=True))
    assert outcome.status is EnrichStatus.UNAVAILABLE


def test_init_db_script(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(PROJECT_ROOT / "scripts/init_db.py"), run_name="__main__")
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert f"schema version {SCHEMA_VERSION}" in out and "Seed data valid" in out
    assert (tmp_path / "app.db").exists()  # DB_PATH points into tmp_path (conftest)
