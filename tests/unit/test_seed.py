"""AC1.2: curated seed files validate, meet size targets and carry honest provenance."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from dubai_planner.config import PROJECT_ROOT
from dubai_planner.data import seed
from dubai_planner.data.seed import SEED_FILES, SeedCatalog, SeedDataError, load_catalog, load_meta
from dubai_planner.models import UAE_BBOX, DataKind, DietaryFlag, PriceRange

SEED_DIR = PROJECT_ROOT / "data" / "seed"


@pytest.fixture(scope="module")
def catalog() -> SeedCatalog:
    return load_catalog(SEED_DIR)


def test_size_targets(catalog: SeedCatalog) -> None:
    assert len(catalog.attractions) >= 50
    assert len(catalog.hotels) >= 30
    assert len({h.area for h in catalog.hotels}) >= 6
    assert len(catalog.restaurants) >= 60


def test_restaurants_have_dietary_flags(catalog: SeedCatalog) -> None:
    flagged = [r for r in catalog.restaurants if r.dietary]
    assert len(flagged) >= 60
    coverage = Counter(flag for r in catalog.restaurants for flag in r.dietary)
    assert coverage[DietaryFlag.VEGETARIAN] >= 30
    assert coverage[DietaryFlag.VEGAN] >= 8
    assert coverage[DietaryFlag.HALAL] >= 25
    assert coverage[DietaryFlag.GLUTEN_FREE] >= 10
    assert sum(r.fully_vegetarian for r in catalog.restaurants) >= 5


def test_every_file_has_meta_with_source_and_date() -> None:
    for filename in SEED_FILES:
        meta = load_meta(filename, SEED_DIR)
        assert meta.source and meta.as_of


def _all_prices(catalog: SeedCatalog) -> list[PriceRange]:
    return (
        [a.price for a in catalog.attractions]
        + [h.nightly_rate for h in catalog.hotels]
        + [r.avg_spend for r in catalog.restaurants]
    )


def test_seed_prices_are_never_labelled_confirmed(catalog: SeedCatalog) -> None:
    """Curated prices are estimates: never 'verified' and always shown with ≈ / estimate."""
    prices = _all_prices(catalog)
    assert all(p.provenance.kind is DataKind.ESTIMATED for p in prices)
    assert all(p.is_estimate for p in prices)
    assert all(p.provenance.note for p in prices)  # carries the "not live-verified" caveat
    t = catalog.transport
    fare_kinds = {
        t.nol_fares.provenance.kind,
        t.taxi.provenance.kind,
        *(f.provenance.kind for f in t.fixed_fares),
    }
    assert fare_kinds == {DataKind.ESTIMATED}


def test_every_record_has_provenance_with_source_and_as_of(catalog: SeedCatalog) -> None:
    records = [
        *catalog.attractions,
        *catalog.hotels,
        *catalog.restaurants,
        *catalog.areas,
        *catalog.events,
        *catalog.transport.metro_stations,
    ]
    for rec in records:
        assert rec.provenance.source and rec.provenance.as_of
        assert rec.provenance.kind is not DataKind.VERIFIED


def test_coordinates_inside_uae(catalog: SeedCatalog) -> None:
    points = [v.location for v in (*catalog.attractions, *catalog.hotels, *catalog.restaurants)]
    points += [a.centroid for a in catalog.areas]
    points += [s.location for s in catalog.transport.metro_stations]
    assert all(p.within(UAE_BBOX) for p in points)


def test_ids_unique_and_prefixed(catalog: SeedCatalog) -> None:
    assert all(a.id.startswith("att_") for a in catalog.attractions)
    assert all(h.id.startswith("htl_") for h in catalog.hotels)
    assert all(r.id.startswith("res_") for r in catalog.restaurants)
    assert len(catalog.known_ids()) == (
        len(catalog.attractions) + len(catalog.hotels) + len(catalog.restaurants)
    )
    assert catalog.venue("att_dubai_frame") is not None
    assert catalog.venue("att_invented_tower") is None


def test_coverage_for_planning_scenarios(catalog: SeedCatalog) -> None:
    """Enough variety for heat, family, day-trip, budget and luxury scenarios."""
    assert sum(a.indoor for a in catalog.attractions) >= 20  # summer heat refuges
    assert sum(a.day_trip for a in catalog.attractions) >= 5
    assert {"budget", "luxury"} <= {h.price_band.value for h in catalog.hotels}
    assert sum(a.price.is_free for a in catalog.attractions) >= 10
    assert any(a.seasonal_months for a in catalog.attractions)


def test_climate_holidays_events_fx(catalog: SeedCatalog) -> None:
    assert catalog.climate.for_month(7).avg_high_c > 38
    assert catalog.climate.for_month(1).avg_high_c < 30
    assert {h.id for h in catalog.holidays.hijri} >= {"hol_eid_al_fitr", "hol_eid_al_adha"}
    assert any(e.id == "evt_dubai_shopping_festival" for e in catalog.events)
    assert catalog.fx_rates.rates["AED"] == Decimal("3.6725")
    assert catalog.fx_rates.rates["USD"] == Decimal("1")
    assert catalog.fx_rates.provenance.kind is DataKind.API  # stored Frankfurter snapshot


def test_transport_helpers(catalog: SeedCatalog) -> None:
    fares = catalog.transport.nol_fares
    assert fares.fare(1) < fares.fare(2) < fares.fare(3) == fares.fare(7)
    assert fares.fare(1, gold=True) > fares.fare(1)
    taxi = catalog.transport.taxi
    assert taxi.estimate(0.5) == taxi.minimum_fare_aed
    assert taxi.estimate(10) > taxi.estimate(5)


# ----------------------------------------------------------------------------- loader errors


def _copy_seed(tmp_path: Path) -> Path:
    dst = tmp_path / "seed"
    shutil.copytree(SEED_DIR, dst)
    return dst


def _edit(path: Path, fn: object) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    fn(doc)  # type: ignore[operator]
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_unknown_area_detected(tmp_path: Path) -> None:
    d = _copy_seed(tmp_path)
    _edit(d / seed.HOTELS, lambda doc: doc["items"][0].update(area="atlantis_city"))
    with pytest.raises(SeedDataError, match="unknown areas"):
        load_catalog(d)


def test_duplicate_ids_detected(tmp_path: Path) -> None:
    d = _copy_seed(tmp_path)
    _edit(d / seed.RESTAURANTS, lambda doc: doc["items"].append(doc["items"][0]))
    with pytest.raises(SeedDataError, match="duplicate ids"):
        load_catalog(d)


def test_invalid_record_rejected(tmp_path: Path) -> None:
    d = _copy_seed(tmp_path)
    _edit(d / seed.ATTRACTIONS, lambda doc: doc["items"][0]["price"].update(min_aed=-5))
    with pytest.raises(ValidationError):
        load_catalog(d)


def test_missing_meta_rejected(tmp_path: Path) -> None:
    d = _copy_seed(tmp_path)
    _edit(d / seed.AREAS, lambda doc: doc.pop("meta"))
    with pytest.raises(SeedDataError, match="meta"):
        load_catalog(d)
