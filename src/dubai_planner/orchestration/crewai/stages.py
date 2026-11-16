"""What each crew stage does: call the shared agent core with inputs from the board.

These are the only places CrewAI mode touches business logic, and they only *call*
it (hard rule 3): the same ``agents.*.run`` functions, validators and revision
strategy as LangGraph mode, injected through :class:`AgentSuite` so tests can stub
any agent in either mode the same way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

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
from dubai_planner.agents.suite import AgentSuite
from dubai_planner.orchestration.crewai.board import CrewRunBoard
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.validation.rules import validate_itinerary

StageFn = Callable[[CrewRunBoard], BaseModel]


@dataclass
class StageRunner:
    """Agent-core calls per stage. ``llm_for(agent)`` gives the agent's chat client."""

    tools: Toolbox
    llm_for: Callable[[str], Any]
    agents: AgentSuite = field(default_factory=AgentSuite)

    def fn(self, stage: str) -> StageFn:
        return {
            "requirements": self.requirements, "attractions": self.attractions,
            "accommodation": self.accommodation, "dining": self.dining,
            "transport": self.transport, "weather_events": self.weather_events,
            "budget": self.budget, "compose": self.compose, "review": self.review,
        }[stage]

    def requirements(self, board: CrewRunBoard) -> BaseModel:
        result: BaseModel = self.agents.requirements(
            requirements.RequirementsInput(message=board.user_message, history=board.history,
                                           previous=board.previous_draft, today=board.today),
            self.llm_for(requirements.AGENT), self.tools)
        return result

    def attractions(self, board: CrewRunBoard) -> BaseModel:
        result: BaseModel = self.agents.attractions(
            attractions.AttractionsInput(request=board.request),
            self.llm_for(attractions.AGENT), self.tools)
        return result

    def accommodation(self, board: CrewRunBoard) -> BaseModel:
        result: BaseModel = self.agents.accommodation(
            accommodation.AccommodationInput(request=board.request),
            self.llm_for(accommodation.AGENT), self.tools)
        return result

    def dining(self, board: CrewRunBoard) -> BaseModel:
        result: BaseModel = self.agents.dining(
            dining.DiningInput(request=board.request), self.llm_for(dining.AGENT), self.tools)
        return result

    def transport(self, board: CrewRunBoard) -> BaseModel:
        result: BaseModel = self.agents.transport(
            transport.TransportInput(request=board.request), None, self.tools)
        return result

    def weather_events(self, board: CrewRunBoard) -> BaseModel:
        req = board.request
        result: BaseModel = self.agents.weather_events(
            weather_events.WeatherEventsInput(start_date=req.start_date, end_date=req.end_date),
            None, self.tools)
        return result

    def budget(self, board: CrewRunBoard) -> BaseModel:
        result: BaseModel = self.agents.budget(
            budget.BudgetInput(request=board.request, research=board.research()),
            self.llm_for(budget.AGENT), self.tools)
        return result

    def compose(self, board: CrewRunBoard) -> BaseModel:
        locked = list(board.locked_days)
        result: BaseModel = self.agents.composer(
            composer.ComposerInput(
                session_id=board.session_id, mode=board.mode, request=board.request,
                research=board.research(), budget=board.budget_plan(),
                feedback=list(board.feedback),
                previous=board.itinerary if locked else None, locked_days=locked),
            self.llm_for(composer.AGENT), self.tools)
        return result

    def review(self, board: CrewRunBoard) -> BaseModel:
        """Deterministic rules first (recorded on the board), then the reviewer agent."""
        itinerary = board.itinerary
        if itinerary is None:
            raise RuntimeError("there is no itinerary to review")
        violations = validate_itinerary(itinerary, board.request, board.research(),
                                        catalog=self.tools.catalog)
        board.violations = violations
        result: BaseModel = self.agents.reviewer(
            reviewer.ReviewerInput(itinerary=itinerary, request=board.request,
                                   violations=violations, iteration=board.iteration),
            self.llm_for(reviewer.AGENT), self.tools)
        return result
