"""Bundle of research-stage outputs (the five parallel research agents)."""

from __future__ import annotations

from pydantic import Field

from dubai_planner.models.common import StrictModel
from dubai_planner.models.context import EventInfo, WeatherContext
from dubai_planner.models.options import AttractionOption, DiningOption, HotelOption
from dubai_planner.models.transport import TransportPlan


class ResearchBundle(StrictModel):
    """Every field is independent so parallel branches can fill them without collisions."""

    attractions: list[AttractionOption] = Field(default_factory=list)
    hotels: list[HotelOption] = Field(default_factory=list)
    dining: list[DiningOption] = Field(default_factory=list)
    transport: TransportPlan | None = None
    weather: WeatherContext | None = None
    events: list[EventInfo] = Field(default_factory=list)

    def known_ids(self) -> set[str]:
        """Ids the composer may reference (grounding)."""
        return (
            {a.id for a in self.attractions}
            | {h.id for h in self.hotels}
            | {d.id for d in self.dining}
            | {e.id for e in self.events}
        )
