"""The set of agent-core functions an orchestrator calls (shared by both modes).

LangGraph nodes and CrewAI tools both call ``suite.<agent>(input, llm, tools)``. Tests
inject stubs here (e.g. ``AgentSuite(composer=spy)``) instead of patching modules, so
the same fault can be injected into either orchestration mode.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from dubai_planner.agents import (
    accommodation,
    attractions,
    budget,
    composer,
    dining,
    requirements,
    reviewer,
    transport,
    weather_events,
)

Agent = Callable[..., Any]


@dataclass(frozen=True)
class AgentSuite:
    """The agent core functions: ``(input, llm, tools) -> output``."""

    requirements: Agent = requirements.run
    attractions: Agent = attractions.run
    accommodation: Agent = accommodation.run
    dining: Agent = dining.run
    transport: Agent = transport.run
    weather_events: Agent = weather_events.run
    budget: Agent = budget.run
    composer: Agent = composer.run
    reviewer: Agent = reviewer.run

    @classmethod
    def default(cls, *, fallback: bool = False) -> AgentSuite:
        """Real agents. ``fallback=True`` opts into the deterministic baselines (Phase 2)."""
        if not fallback:
            return cls()
        return cls(attractions=partial(attractions.run, fallback=True),
                   accommodation=partial(accommodation.run, fallback=True),
                   dining=partial(dining.run, fallback=True),
                   composer=partial(composer.run, fallback=True),
                   reviewer=partial(reviewer.run, fallback=True))
