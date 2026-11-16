"""The tool registry: plain typed functions bound to injectable data sources.

Agents receive a :class:`Toolbox` (``run(input, llm, tools)``). Every method is
deterministic Python over curated seed data or a cached public API with an offline
fallback, so no tool can raise because the network is down. ``TOOL_SPECS`` describes
the LLM-callable tools with Pydantic argument models; Phase 4 wraps these as CrewAI
tools, and :meth:`Toolbox.call` gives a validated, name-based entry point.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from dubai_planner.config import get_settings
from dubai_planner.data import calendar
from dubai_planner.data.fx import FxClient, convert
from dubai_planner.data.open_meteo import OpenMeteoClient
from dubai_planner.data.overpass import OverpassClient, PoiCategory
from dubai_planner.data.seed import SeedCatalog, load_catalog
from dubai_planner.data.wikivoyage import WikivoyageClient
from dubai_planner.models.budget import CostBreakdown
from dubai_planner.models.common import GeoPoint
from dubai_planner.models.context import EventInfo, WeatherContext
from dubai_planner.models.itinerary import DayPlan
from dubai_planner.models.options import AttractionOption, DiningOption, HotelOption
from dubai_planner.models.request import Travellers, TripRequest
from dubai_planner.models.sources import OsmPoi, WikiSummary
from dubai_planner.models.transport import TransportLeg, TransportMode
from dubai_planner.tools import costs, search, travel


@dataclass
class Toolbox:
    catalog: SeedCatalog
    weather_client: OpenMeteoClient = field(default_factory=OpenMeteoClient)
    fx_client: FxClient = field(default_factory=FxClient)
    overpass_client: OverpassClient = field(default_factory=OverpassClient)
    wikivoyage_client: WikivoyageClient = field(default_factory=WikivoyageClient)

    @classmethod
    def default(cls) -> Toolbox:
        return cls(catalog=load_catalog())

    # ------------------------------------------------------------------ search
    def search_attractions(self, request: TripRequest, limit: int = 25) -> list[AttractionOption]:
        return search.search_attractions(request, self.catalog, limit=limit)

    def search_hotels(self, request: TripRequest, limit: int = 12) -> list[HotelOption]:
        return search.search_hotels(request, self.catalog, limit=limit)

    def search_restaurants(self, request: TripRequest, limit: int = 25) -> list[DiningOption]:
        return search.search_restaurants(request, self.catalog, limit=limit)

    # ------------------------------------------------------------------ context
    def get_weather_context(self, start: date, end: date) -> WeatherContext:
        return self.weather_client.get_weather_context(start, end)

    def get_events(self, start: date, end: date) -> list[EventInfo]:
        return calendar.get_events(start, end)

    def find_osm_pois(self, category: PoiCategory) -> list[OsmPoi]:
        return self.overpass_client.search_pois(category)

    def get_area_guide(self, title: str) -> WikiSummary | None:
        return self.wikivoyage_client.get_summary(title)

    # ------------------------------------------------------------------ travel
    def location_of(self, ref: str) -> GeoPoint | None:
        """Coordinates of a venue id or an area id."""
        venue = self.catalog.venue(ref)
        if venue is not None:
            return venue.location
        area = self.catalog.area(ref)
        return area.centroid if area is not None else None

    def estimate_travel(
        self,
        origin: GeoPoint,
        destination: GeoPoint,
        *,
        travellers: Travellers | None = None,
        mode: TransportMode | None = None,
        prefer_taxi: bool = False,
        from_ref: str | None = None,
        to_ref: str | None = None,
    ) -> TransportLeg:
        return travel.estimate_travel(
            origin, destination, self.catalog.transport, travellers=travellers, mode=mode,
            prefer_taxi=prefer_taxi, from_ref=from_ref, to_ref=to_ref,
        )

    def estimate_travel_between(
        self, from_ref: str, to_ref: str, *, travellers: Travellers | None = None,
        mode: TransportMode | None = None,
    ) -> TransportLeg:
        origin, destination = self.location_of(from_ref), self.location_of(to_ref)
        if origin is None or destination is None:
            missing = from_ref if origin is None else to_ref
            raise ValueError(f"unknown venue or area id {missing!r}")
        return self.estimate_travel(origin, destination, travellers=travellers, mode=mode,
                                    from_ref=from_ref, to_ref=to_ref)

    def metro_fare(self, zones: int, gold: bool = False) -> Decimal:
        return travel.metro_fare(self.catalog.transport, zones, gold=gold)

    # ------------------------------------------------------------------ money
    def convert_currency(self, amount: Decimal | float | int | str, from_currency: str,
                         to_currency: str = "AED") -> Decimal:
        return convert(amount, from_currency, to_currency, client=self.fx_client)

    def calculate_costs(
        self,
        days: Sequence[DayPlan],
        *,
        hotel: HotelOption | None,
        request: TripRequest,
        contingency_pct: Decimal,
    ) -> CostBreakdown:
        return costs.calculate_costs(
            days, hotel=hotel, nights=request.nights, travellers=request.travellers,
            budget_aed=request.total_budget_aed(), contingency_pct=contingency_pct,
            tolerance=get_settings().budget_tolerance,
        )

    # ------------------------------------------------------------------ registry
    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Validate ``arguments`` against the tool's model and run it (CrewAI/LLM entry)."""
        spec = TOOL_SPECS.get(name)
        if spec is None:
            raise KeyError(f"unknown tool {name!r}; available: {sorted(TOOL_SPECS)}")
        return spec.handler(self, spec.args_model.model_validate(arguments))


# --------------------------------------------------------------------------- specs


class SearchArgs(BaseModel):
    request: TripRequest
    limit: int = Field(default=20, ge=1, le=60)


class DateRangeArgs(BaseModel):
    start: date
    end: date


class TravelArgs(BaseModel):
    from_ref: str = Field(description="Venue id (att_/htl_/res_) or area id")
    to_ref: str = Field(description="Venue id (att_/htl_/res_) or area id")
    adults: int = Field(default=1, ge=1, le=20)
    children: int = Field(default=0, ge=0, le=20)
    mode: TransportMode | None = None


class MetroFareArgs(BaseModel):
    zones: int = Field(ge=1, le=3)
    gold: bool = False


class ConvertArgs(BaseModel):
    amount: Decimal = Field(ge=0)
    from_currency: str = Field(min_length=3, max_length=3)
    to_currency: str = Field(default="AED", min_length=3, max_length=3)


class CostArgs(BaseModel):
    request: TripRequest
    days: list[DayPlan]
    hotel_id: str | None = None
    contingency_pct: Decimal = Field(default=Decimal("0.10"), ge=0, le=Decimal("0.5"))


def _cost_handler(tb: Toolbox, a: CostArgs) -> CostBreakdown:
    venue = tb.catalog.venue(a.hotel_id) if a.hotel_id else None
    if a.hotel_id and not isinstance(venue, HotelOption):
        raise ValueError(f"unknown hotel id {a.hotel_id!r}")
    hotel = venue if isinstance(venue, HotelOption) else None
    return tb.calculate_costs(a.days, hotel=hotel, request=a.request,
                              contingency_pct=a.contingency_pct)


def _travel_handler(tb: Toolbox, a: TravelArgs) -> TransportLeg:
    party = Travellers(adults=a.adults, children_ages=[10] * a.children)
    return tb.estimate_travel_between(a.from_ref, a.to_ref, travellers=party, mode=a.mode)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Toolbox, Any], Any]


TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec("search_attractions", "Rank curated Dubai attractions for a trip request.",
                 SearchArgs, lambda tb, a: tb.search_attractions(a.request, a.limit)),
        ToolSpec("search_hotels", "Rank curated hotels that fit the request's budget.",
                 SearchArgs, lambda tb, a: tb.search_hotels(a.request, a.limit)),
        ToolSpec("search_restaurants", "Restaurants meeting dietary needs, ranked by fit.",
                 SearchArgs, lambda tb, a: tb.search_restaurants(a.request, a.limit)),
        ToolSpec("get_weather_context", "Daily weather (forecast/historical/climatology).",
                 DateRangeArgs, lambda tb, a: tb.get_weather_context(a.start, a.end)),
        ToolSpec("get_events", "Events, public holidays and Ramadan overlapping dates.",
                 DateRangeArgs, lambda tb, a: tb.get_events(a.start, a.end)),
        ToolSpec("estimate_travel", "Estimated travel leg (mode, minutes, AED) between ids.",
                 TravelArgs, _travel_handler),
        ToolSpec("metro_fare", "Dubai Metro single-journey Nol fare per person (AED).",
                 MetroFareArgs, lambda tb, a: tb.metro_fare(a.zones, a.gold)),
        ToolSpec("convert_currency", "Convert money; AED via the USD peg 3.6725.",
                 ConvertArgs,
                 lambda tb, a: tb.convert_currency(a.amount, a.from_currency, a.to_currency)),
        ToolSpec("calculate_costs", "Itemised cost breakdown (AED) for planned days.",
                 CostArgs, _cost_handler),
    ]
}
