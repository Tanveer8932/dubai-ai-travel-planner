"""Typed results of the public-data source clients (FX, geocoding, OSM POIs, wiki text)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import Field, field_validator

from dubai_planner.models.common import GeoPoint, Provenance, StrictModel, to_decimal


class SeedMeta(StrictModel):
    """Header of every ``data/seed/*.json`` file."""

    description: str
    source: str
    as_of: date
    licence: str | None = None
    notes: list[str] = Field(default_factory=list)


class FxRates(StrictModel):
    """Units of each currency per 1 USD (USD is the hub because AED is pegged to it)."""

    base: Literal["USD"] = "USD"
    rates: dict[str, Decimal]
    provenance: Provenance

    @field_validator("rates", mode="before")
    @classmethod
    def _to_decimal(cls, value: object) -> object:
        if isinstance(value, dict):
            return {str(k).upper(): to_decimal(v) for k, v in value.items()}
        return value

    @field_validator("rates")
    @classmethod
    def _positive(cls, value: dict[str, Decimal]) -> dict[str, Decimal]:
        if any(rate <= 0 for rate in value.values()):
            raise ValueError("FX rates must be positive")
        return value


class GeocodeResult(StrictModel):
    query: str
    name: str
    location: GeoPoint
    display_name: str = ""
    osm_id: str | None = None
    category: str | None = None
    extratags: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance


class OsmPoi(StrictModel):
    osm_id: str
    name: str
    location: GeoPoint
    tags: dict[str, str] = Field(default_factory=dict)
    ref_id: str | None = Field(default=None, description="Seed id when derived from seed")
    provenance: Provenance


class WikiSummary(StrictModel):
    title: str
    extract: str
    url: str | None = None
    provenance: Provenance
