"""The shared contracts must work unchanged in both orchestrators (Phase 3 / Phase 4)."""

from __future__ import annotations

import pytest
from factories import itinerary, trip_request

from dubai_planner.models import Itinerary, TripRequest


def test_langgraph_checkpoint_serializer_round_trip() -> None:
    """Pydantic values in LangGraph state must survive the SqliteSaver serializer."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    serde = JsonPlusSerializer()
    state = {"request": trip_request(), "itinerary": itinerary()}
    restored = serde.loads_typed(serde.dumps_typed(state))
    # Depending on the serializer's allow-list, models come back as models or as dicts;
    # either way they must re-validate into identical contracts.
    assert TripRequest.model_validate(restored["request"]) == state["request"]
    assert Itinerary.model_validate(restored["itinerary"]) == state["itinerary"]


def test_crewai_task_accepts_output_pydantic() -> None:
    """CrewAI tasks take our models directly as ``output_pydantic``."""
    pytest.importorskip("crewai")
    from crewai import Task

    task = Task(
        description="Compose the itinerary",
        expected_output="An Itinerary",
        output_pydantic=Itinerary,
    )
    assert task.output_pydantic is Itinerary
