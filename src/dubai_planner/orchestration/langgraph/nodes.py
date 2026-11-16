"""LangGraph nodes: thin wrappers that call the shared agent core (hard rule 3).

Every node

- runs inside the run's trace context (``run_id``, ``mode="langgraph"``, ``iteration``),
  set from the state itself, so it is correct in LangGraph's worker threads and after
  a resume in a new process;
- emits ``start``/``end`` (or ``error``) :class:`TraceEvent`s named after the node, with
  ``payload["kind"] == "node"`` (the agents' own spans are nested inside);
- returns only the state keys it owns.

The agent callables are injectable (:class:`AgentSuite`) so tests can stub them, e.g. with
fixed delays to prove parallelism, without touching the agent core.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

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
from dubai_planner.agents.revision import applicable_locks, plan_revision
from dubai_planner.agents.suite import AgentSuite
from dubai_planner.llm import get_chat_llm
from dubai_planner.models import (
    Attempt,
    ChatMessage,
    ChatRole,
    Itinerary,
    PlanEdit,
    PlannerMode,
    PlanStatus,
    Severity,
    TraceEventType,
)
from dubai_planner.orchestration.common import (
    DEFAULT_QUESTION,
    attempt_summary,
    better,
    summary_text,
)
from dubai_planner.orchestration.langgraph.state import PlannerState
from dubai_planner.storage import Repository
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import agent_span, emit, run_context
from dubai_planner.validation.rules import validate_itinerary

MODE = PlannerMode.LANGGRAPH

# Node names (also used by the graph builder, streaming and the UI).
PARSE = "parse_request"
CLARIFY = "ask_clarification"
RESEARCH_ATTRACTIONS = "research_attractions"
RESEARCH_ACCOMMODATION = "research_accommodation"
RESEARCH_DINING = "research_dining"
RESEARCH_TRANSPORT = "research_transport"
RESEARCH_WEATHER = "research_weather_events"
RESEARCH_NODES = (RESEARCH_ATTRACTIONS, RESEARCH_ACCOMMODATION, RESEARCH_DINING,
                  RESEARCH_TRANSPORT, RESEARCH_WEATHER)
MERGE = "merge_research"
BUDGET = "allocate_budget"
COMPOSE = "compose_itinerary"
VALIDATE = "validate"
REVIEW = "review"
REFINE = "refine_itinerary"
FINALIZE = "finalize"

def _llm_from_factory() -> Callable[[str], Any]:
    cache: dict[str, Any] = {}

    def llm_for(agent: str) -> Any:
        if agent not in cache:
            cache[agent] = get_chat_llm(agent)  # honours LLM_MODEL__<AGENT>
        return cache[agent]

    return llm_for


@dataclass
class PlannerDeps:
    """Everything the nodes need besides state. ``llm=None`` → provider from settings."""

    tools: Toolbox
    llm: Any = None
    agents: AgentSuite = field(default_factory=AgentSuite)
    repo: Repository | None = None
    max_review_iterations: int = 2
    max_clarifications: int = 3
    _llm_for: Callable[[str], Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        fixed = self.llm
        self._llm_for = (lambda _agent: fixed) if fixed is not None else _llm_from_factory()

    def llm_for(self, agent: str) -> Any:
        return self._llm_for(agent)


@contextmanager
def node_span(name: str, state: PlannerState) -> Iterator[dict[str, Any]]:
    """Run context from state + node start/end events. Interrupts are not errors."""
    with run_context(state.get("run_id") or "adhoc", mode=MODE.value,
                     iteration=state.get("iteration", 0)):
        try:
            with agent_span(name, kind="node") as span:
                span["kind"] = "node"
                yield span
        except GraphBubbleUp:  # interrupt(): the node pauses, it did not fail
            emit(name, TraceEventType.INFO, payload={"kind": "node", "interrupted": True})
            raise


def _message(role: ChatRole, text: str, **meta: Any) -> ChatMessage:
    return ChatMessage(role=role, content=text[:20_000], metadata=meta)


class PlannerNodes:
    """Node callables bound to their dependencies."""

    def __init__(self, deps: PlannerDeps) -> None:
        self.deps = deps

    # ------------------------------------------------------------------ requirements
    def parse_request(self, state: PlannerState) -> dict[str, Any]:
        with node_span(PARSE, state) as span:
            history = list(state.get("messages", []))[:-1]  # the last one is this message
            fixed = state.get("fixed_request")
            reused = (requirements.unchanged(PlanEdit(base_request=fixed, keep_request=True))
                      if fixed is not None else None)
            result = reused or self.deps.agents.requirements(
                requirements.RequirementsInput(
                    message=state["user_message"], history=history,
                    previous=state.get("draft"), today=state["today"]),
                self.deps.llm_for(requirements.AGENT), self.deps.tools)
            span.update(complete=result.is_complete, missing=result.missing)
            return {
                "draft": result.draft, "request": result.request, "missing": result.missing,
                "clarification_question": result.clarification_question,
                "assumptions": [*state.get("assumptions", []), *result.assumptions],
            }

    def ask_clarification(self, state: PlannerState) -> dict[str, Any]:
        with node_span(CLARIFY, state) as span:
            question = state.get("clarification_question") or DEFAULT_QUESTION
            answer = interrupt({"question": question, "missing": state.get("missing", [])})
            text = str(answer).strip()
            if not text:
                raise ValueError("the clarification answer is empty")
            span.update(answered=True)
            return {
                "user_message": text,
                "clarifications": state.get("clarifications", 0) + 1,
                "messages": [_message(ChatRole.ASSISTANT, question, kind="clarification"),
                             _message(ChatRole.USER, text, kind="clarification_answer")],
            }

    # ------------------------------------------------------------------ research (parallel)
    def _request(self, state: PlannerState) -> Any:
        request = state.get("request")
        if request is None:  # pragma: no cover - guarded by routing
            raise RuntimeError("research needs a complete request")
        return request

    def _itinerary(self, state: PlannerState) -> Itinerary:
        itinerary = state.get("itinerary")
        if itinerary is None:  # pragma: no cover - guarded by the graph's edges
            raise RuntimeError("no itinerary to check")
        return itinerary

    def research_attractions(self, state: PlannerState) -> dict[str, Any]:
        with node_span(RESEARCH_ATTRACTIONS, state) as span:
            out = self.deps.agents.attractions(
                attractions.AttractionsInput(request=self._request(state)),
                self.deps.llm_for(attractions.AGENT), self.deps.tools)
            span.update(options=len(out.options))
            return {"research": {"attractions": out.options}}

    def research_accommodation(self, state: PlannerState) -> dict[str, Any]:
        with node_span(RESEARCH_ACCOMMODATION, state) as span:
            out = self.deps.agents.accommodation(
                accommodation.AccommodationInput(request=self._request(state)),
                self.deps.llm_for(accommodation.AGENT), self.deps.tools)
            span.update(options=len(out.options))
            return {"research": {"hotels": out.options}}

    def research_dining(self, state: PlannerState) -> dict[str, Any]:
        with node_span(RESEARCH_DINING, state) as span:
            out = self.deps.agents.dining(dining.DiningInput(request=self._request(state)),
                                          self.deps.llm_for(dining.AGENT), self.deps.tools)
            span.update(options=len(out.options))
            return {"research": {"dining": out.options}}

    def research_transport(self, state: PlannerState) -> dict[str, Any]:
        with node_span(RESEARCH_TRANSPORT, state):
            plan = self.deps.agents.transport(
                transport.TransportInput(request=self._request(state)), None, self.deps.tools)
            return {"research": {"transport": plan}}

    def research_weather_events(self, state: PlannerState) -> dict[str, Any]:
        with node_span(RESEARCH_WEATHER, state) as span:
            req = self._request(state)
            out = self.deps.agents.weather_events(
                weather_events.WeatherEventsInput(start_date=req.start_date,
                                                  end_date=req.end_date),
                None, self.deps.tools)
            span.update(events=len(out.events))
            return {"research": {"weather": out.weather, "events": out.events}}

    def merge_research(self, state: PlannerState) -> dict[str, Any]:
        """Fan-in: the reducer has already merged the branches; check nothing is missing."""
        with node_span(MERGE, state) as span:
            research = state.get("research")
            if research is None or research.transport is None or research.weather is None:
                raise RuntimeError("research fan-in is incomplete")
            span.update(attractions=len(research.attractions), hotels=len(research.hotels),
                        dining=len(research.dining), events=len(research.events))
            return {}

    # ------------------------------------------------------------------ planning
    def allocate_budget(self, state: PlannerState) -> dict[str, Any]:
        with node_span(BUDGET, state) as span:
            plan = self.deps.agents.budget(
                budget.BudgetInput(request=self._request(state), research=state["research"]),
                self.deps.llm_for(budget.AGENT), self.deps.tools)
            span.update(feasible=plan.feasible)
            return {"budget_plan": plan}

    def compose_itinerary(self, state: PlannerState) -> dict[str, Any]:
        with node_span(COMPOSE, state) as span:
            request, previous = self._request(state), state.get("itinerary")
            locked = applicable_locks(list(state.get("locked_days", [])), previous, request)
            itinerary = self.deps.agents.composer(
                composer.ComposerInput(
                    session_id=state["session_id"], mode=MODE, request=request,
                    research=state["research"], budget=state["budget_plan"],
                    feedback=list(state.get("feedback", [])),
                    previous=previous if locked else None, locked_days=locked,
                    day_constraints=list(state.get("day_constraints", []))),
                self.deps.llm_for(composer.AGENT), self.deps.tools)
            span.update(total=str(itinerary.cost.total), locked_days=locked)
            return {"itinerary": itinerary}

    def validate(self, state: PlannerState) -> dict[str, Any]:
        with node_span(VALIDATE, state) as span:
            violations = validate_itinerary(self._itinerary(state), self._request(state),
                                            state.get("research"),
                                            catalog=self.deps.tools.catalog)
            span.update(errors=sum(v.severity is Severity.ERROR for v in violations),
                        violations=len(violations))
            return {"validation": violations}

    def review(self, state: PlannerState) -> dict[str, Any]:
        with node_span(REVIEW, state) as span:
            iteration = state.get("iteration", 0)
            itinerary = self._itinerary(state)
            violations = list(state.get("validation", []))
            report = self.deps.agents.reviewer(
                reviewer.ReviewerInput(itinerary=itinerary, request=self._request(state),
                                       violations=violations, iteration=iteration),
                self.deps.llm_for(reviewer.AGENT), self.deps.tools)
            attempt = Attempt(iteration=iteration, itinerary=itinerary, review=report,
                              violations=violations)
            best = better(attempt, state.get("best"))
            summary = attempt_summary(attempt, list(state.get("locked_days", [])))
            span.update(approved=attempt.approved, score=report.score, errors=attempt.errors)
            return {"review": report, "best": best,
                    "attempts": [*state.get("attempts", []), summary]}

    def refine_itinerary(self, state: PlannerState) -> dict[str, Any]:
        """Targeted fix: feedback for the composer and the days to keep unchanged."""
        with node_span(REFINE, state) as span:
            plan = plan_revision(state["review"], state["itinerary"],  # type: ignore[arg-type]
                                 num_days=self._request(state).num_days)
            span.update(target_days=plan.target_days, locked_days=plan.locked_days)
            return {"iteration": state.get("iteration", 0) + 1, "feedback": plan.feedback,
                    "locked_days": plan.locked_days}

    # ------------------------------------------------------------------ end
    def finalize(self, state: PlannerState) -> dict[str, Any]:
        with node_span(FINALIZE, state) as span:
            best = state.get("best")
            request = state.get("request")
            iterations = state.get("iteration", 0)
            if request is None or best is None:
                status = PlanStatus.NEEDS_CLARIFICATION
            elif best.approved:  # reviewer approval AND no error-severity rule violation
                status = PlanStatus.APPROVED
            else:
                status = PlanStatus.NEEDS_REVISION
            version = self._persist(state, best, status) if best is not None else None
            text = summary_text(status, best, state.get("clarification_question"), iterations)
            span.update(status=status.value, saved_version=version)
            update: dict[str, Any] = {
                "status": status, "saved_version": version,
                "messages": [_message(ChatRole.ASSISTANT, text, kind="plan_result",
                                      status=status.value, run_id=state.get("run_id"))],
            }
            if best is not None:
                update |= {"itinerary": best.itinerary, "review": best.review,
                           "validation": best.violations}
            return update

    def _persist(self, state: PlannerState, best: Attempt, status: PlanStatus) -> int | None:
        """Save the returned itinerary as the session's next version (once per run)."""
        repo = self.deps.repo
        if repo is None:
            return None
        session_id, run_id = state["session_id"], state.get("run_id")
        for record in reversed(repo.list_itineraries(session_id)):  # idempotent on resume
            if run_id and record.run_id == run_id:
                return record.version
        if repo.get_session(session_id) is None:  # graph driven without the service
            repo.create_session(title="Dubai trip", mode=MODE, session_id=session_id)
        if run_id and repo.get_run(run_id) is None:
            repo.start_run(MODE, session_id=session_id, run_id=run_id)
        record = repo.save_itinerary(best.itinerary, request=state.get("request"),
                                     approved=status is PlanStatus.APPROVED, run_id=run_id)
        return record.version
