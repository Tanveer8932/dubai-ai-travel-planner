"""Load and validate the curated seed data in ``SEED_DIR`` (default ``data/seed``).

Every file is a JSON object ``{"meta": SeedMeta, "items": [...]}`` (list files) or
``{"meta": SeedMeta, "data": {...}}`` (single-object files). Each record carries its own
``provenance``; prices are always ``estimated``. Loading fails loudly on invalid data
(``pydantic.ValidationError`` / ``SeedDataError``) so bad seed data is caught by tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter

from dubai_planner.config import get_settings
from dubai_planner.models.context import ClimateNormals, HolidaySeed, RecurringEvent
from dubai_planner.models.options import Area, AttractionOption, DiningOption, HotelOption
from dubai_planner.models.sources import FxRates, SeedMeta
from dubai_planner.models.transport import TransportSeed

T = TypeVar("T", bound=BaseModel)

ATTRACTIONS = "attractions.json"
HOTELS = "hotels.json"
RESTAURANTS = "restaurants.json"
AREAS = "areas.json"
TRANSPORT = "transport.json"
EVENTS = "events.json"
CLIMATE = "climate.json"
HOLIDAYS = "holidays.json"
FX_RATES = "fx_rates.json"

SEED_FILES = (ATTRACTIONS, HOTELS, RESTAURANTS, AREAS, TRANSPORT, EVENTS, CLIMATE, HOLIDAYS,
              FX_RATES)


class SeedDataError(ValueError):
    """Seed data is structurally valid JSON but violates a cross-record invariant."""


def _seed_dir(seed_dir: Path | None) -> Path:
    return seed_dir if seed_dir is not None else get_settings().seed_dir


def _read(seed_dir: Path, filename: str) -> dict[str, Any]:
    path = seed_dir / filename
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "meta" not in raw:
        raise SeedDataError(f"{filename}: expected an object with a 'meta' key")
    SeedMeta.model_validate(raw["meta"])
    return raw


def _load_items(seed_dir: Path, filename: str, model: type[T]) -> list[T]:
    raw = _read(seed_dir, filename)
    items = TypeAdapter(list[model]).validate_python(raw.get("items"))  # type: ignore[valid-type]
    ids = [getattr(item, "id", None) for item in items]
    dupes = sorted({i for i in ids if i is not None and ids.count(i) > 1})
    if dupes:
        raise SeedDataError(f"{filename}: duplicate ids {dupes}")
    return items


def _load_data(seed_dir: Path, filename: str, model: type[T]) -> T:
    return model.model_validate(_read(seed_dir, filename).get("data"))


def load_meta(filename: str, seed_dir: Path | None = None) -> SeedMeta:
    return SeedMeta.model_validate(_read(_seed_dir(seed_dir), filename)["meta"])


@dataclass(frozen=True)
class SeedCatalog:
    """All curated data, validated and cross-checked, with id lookups."""

    attractions: list[AttractionOption]
    hotels: list[HotelOption]
    restaurants: list[DiningOption]
    areas: list[Area]
    transport: TransportSeed
    events: list[RecurringEvent]
    climate: ClimateNormals
    holidays: HolidaySeed
    fx_rates: FxRates
    _by_id: dict[str, AttractionOption | HotelOption | DiningOption] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        index: dict[str, AttractionOption | HotelOption | DiningOption] = {}
        venues: list[AttractionOption | HotelOption | DiningOption] = [
            *self.attractions, *self.hotels, *self.restaurants
        ]
        for venue in venues:
            if venue.id in index:
                raise SeedDataError(f"venue id {venue.id!r} is used more than once")
            index[venue.id] = venue
        self._by_id.update(index)
        area_ids = {a.id for a in self.areas}
        unknown = sorted(
            {f"{v.id}->{v.area}" for v in index.values() if v.area not in area_ids}
            | {f"{s.id}->{s.area}" for s in self.transport.metro_stations
               if s.area is not None and s.area not in area_ids}
        )
        if unknown:
            raise SeedDataError(f"records reference unknown areas: {unknown}")

    def venue(self, venue_id: str) -> AttractionOption | HotelOption | DiningOption | None:
        return self._by_id.get(venue_id)

    def area(self, area_id: str) -> Area | None:
        return next((a for a in self.areas if a.id == area_id), None)

    def known_ids(self) -> set[str]:
        return set(self._by_id)


def load_catalog(seed_dir: Path | None = None) -> SeedCatalog:
    """Load every seed file (cached per directory)."""
    return _load_catalog_cached(_seed_dir(seed_dir).resolve())


@lru_cache(maxsize=4)
def _load_catalog_cached(seed_dir: Path) -> SeedCatalog:
    return SeedCatalog(
        attractions=_load_items(seed_dir, ATTRACTIONS, AttractionOption),
        hotels=_load_items(seed_dir, HOTELS, HotelOption),
        restaurants=_load_items(seed_dir, RESTAURANTS, DiningOption),
        areas=_load_items(seed_dir, AREAS, Area),
        transport=_load_data(seed_dir, TRANSPORT, TransportSeed),
        events=_load_items(seed_dir, EVENTS, RecurringEvent),
        climate=_load_data(seed_dir, CLIMATE, ClimateNormals),
        holidays=_load_data(seed_dir, HOLIDAYS, HolidaySeed),
        fx_rates=_load_data(seed_dir, FX_RATES, FxRates),
    )


def clear_cache() -> None:
    _load_catalog_cached.cache_clear()
