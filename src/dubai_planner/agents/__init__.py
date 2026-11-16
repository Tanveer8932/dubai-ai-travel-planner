"""Framework-agnostic agent core (architecture §5).

Each module exposes ``AGENT``, Pydantic input/output models and
``run(input, llm, tools, ...)``. LangGraph nodes (Phase 3) and CrewAI tasks (Phase 4)
call these same functions; no agent logic is duplicated per framework.

| Agent | Module | LLM |
|-------|--------|-----|
| Requirements | ``requirements`` | extraction |
| Attractions | ``attractions`` | pick + explain (grounded ids) |
| Accommodation | ``accommodation`` | pick + explain (grounded ids) |
| Dining | ``dining`` | pick + explain (grounded ids) |
| Transport | ``transport`` | none (deterministic) |
| Weather & events | ``weather_events`` | none (deterministic) |
| Budget | ``budget`` | rationale only (allocation is deterministic) |
| Itinerary composer | ``composer`` | draft schedule (Python grounds and prices it) |
| Reviewer | ``reviewer`` | quality review (approval decided by rules) |

Importing this package registers each agent's deterministic mock reply.
"""

from __future__ import annotations

from dubai_planner.agents import (
    accommodation,
    attractions,
    budget,
    composer,
    dining,
    requirements,
    reviewer,
    selection,
    transport,
    weather_events,
)

AGENT_NAMES = (
    requirements.AGENT,
    attractions.AGENT,
    accommodation.AGENT,
    dining.AGENT,
    transport.AGENT,
    weather_events.AGENT,
    budget.AGENT,
    composer.AGENT,
    reviewer.AGENT,
)

__all__ = [
    "AGENT_NAMES",
    "accommodation",
    "attractions",
    "budget",
    "composer",
    "dining",
    "requirements",
    "reviewer",
    "selection",
    "transport",
    "weather_events",
]
