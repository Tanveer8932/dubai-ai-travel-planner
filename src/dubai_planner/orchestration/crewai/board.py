"""Per-run shared state of a CrewAI planning run (the crew's "blackboard").

CrewAI passes task results between agents as *text* (task ``context``). Our contracts
are typed Pydantic models and the maths must stay deterministic, so the typed results
travel on this board instead:

1. a stage tool runs the shared agent core and puts its typed output here as
   *pending*, returning only a short text brief to the CrewAI agent;
2. the task's guardrail takes the pending output (rejecting the task if the agent
   never called its tool, i.e. answered from its own imagination), validates it and
   attaches it to the task as ``TaskOutput.pydantic``;
3. later stages read their inputs from the accepted outputs.

Failures are recorded here too: CrewAI swallows tool exceptions and lets the agent
retry (found while building this adapter), so tools record the first failure, stop
working, and every guardrail re-raises it. Nothing is silently approved.

The board lives only for one run; the session's durable state is in SQLite.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date
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
    weather_events,
)
from dubai_planner.agents.revision import RevisionPlan
from dubai_planner.models import (
    Attempt,
    AttemptSummary,
    BudgetPlan,
    ChatMessage,
    DayConstraint,
    Itinerary,
    PlanEdit,
    PlannerMode,
    ResearchBundle,
    ReviewReport,
    RuleViolation,
    TransportPlan,
    TripRequest,
    TripRequestDraft,
)
from dubai_planner.orchestration.common import attempt_summary, better

FAILED_MARKER = "STAGE_FAILED"
CANCELLED_MARKER = "STAGE_CANCELLED"


class CrewRunAborted(RuntimeError):
    """Raised inside the crew to stop it after a recorded failure or a cancellation."""


class CrewRunCancelled(CrewRunAborted):
    """The caller stopped consuming the stream."""


@dataclass(frozen=True)
class Stage:
    """One unit of crew work: a task, its owning role, its tool and its output model."""

    key: str
    task: str  # task name in tasks.yaml (= trace name; matches the LangGraph node)
    role_key: str  # agent key in agents.yaml
    tool: str  # the role's tool that runs the shared agent core
    core_agent: str | None  # agent-core name (LLM selection via LLM_MODEL__<AGENT>)
    output: type[BaseModel]


STAGES: dict[str, Stage] = {s.key: s for s in [
    Stage("requirements", "parse_request", "requirements_analyst",
          "extract_trip_requirements", requirements.AGENT, requirements.RequirementsResult),
    Stage("attractions", "research_attractions", "attractions_specialist",
          "research_attractions", attractions.AGENT, attractions.AttractionsResult),
    Stage("accommodation", "research_accommodation", "accommodation_specialist",
          "research_accommodation", accommodation.AGENT, accommodation.AccommodationResult),
    Stage("dining", "research_dining", "dining_specialist", "research_dining", dining.AGENT,
          dining.DiningResult),
    Stage("transport", "research_transport", "transport_specialist", "plan_transport",
          None, TransportPlan),
    Stage("weather_events", "research_weather_events", "climate_events_analyst",
          "research_weather_events", None, weather_events.WeatherEventsResult),
    Stage("budget", "allocate_budget", "budget_controller", "allocate_budget",
          budget.AGENT, BudgetPlan),
    Stage("compose", "compose_itinerary", "itinerary_architect", "compose_itinerary",
          composer.AGENT, Itinerary),
    Stage("review", "review_itinerary", "quality_reviewer", "review_itinerary",
          reviewer.AGENT, ReviewReport),
]}
RESEARCH_STAGES = ("attractions", "accommodation", "dining", "transport", "weather_events")
MANAGER_ROLE_KEY = "trip_manager"


@dataclass
class CrewRunBoard:
    """Typed hand-over between crew stages for one run. Thread-safe."""

    session_id: str
    run_id: str
    mode: PlannerMode
    today: date
    user_message: str
    history: list[ChatMessage] = field(default_factory=list)
    previous_draft: TripRequestDraft | None = None
    max_iterations: int = 2
    edit: PlanEdit | None = None  # a refinement of a saved itinerary (Phase 5)
    day_constraints: list[DayConstraint] = field(default_factory=list)
    # --- results
    requirements: requirements.RequirementsResult | None = None
    accepted: dict[str, BaseModel] = field(default_factory=dict)
    itinerary: Itinerary | None = None  # the attempt being checked/reviewed
    violations: list[RuleViolation] = field(default_factory=list)  # of ``itinerary``
    # --- review loop
    iteration: int = 0  # refinements so far (same meaning as LangGraph's ``iteration``)
    feedback: list[str] = field(default_factory=list)
    locked_days: list[int] = field(default_factory=list)
    attempts: list[AttemptSummary] = field(default_factory=list)
    last_attempt: Attempt | None = None
    best: Attempt | None = None
    guardrail_rejections: int = 0
    # --- clarification gate
    clarification_rounds: int = 0
    gave_up: bool = False
    # --- control
    error: BaseException | None = None
    failed_stage: str | None = None
    cancelled: bool = False
    _pending: dict[str, BaseModel] = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ------------------------------------------------------------------ inputs
    @property
    def request(self) -> TripRequest:
        req = self.requirements.request if self.requirements else None
        if req is None:
            raise RuntimeError("planning needs a complete trip request")
        return req

    def research(self) -> ResearchBundle:
        """The research stages' accepted outputs as one bundle (composer/rules input)."""
        with self._lock:
            got = self.accepted
            weather = got.get("weather_events")
            return ResearchBundle(
                attractions=list(getattr(got.get("attractions"), "options", [])),
                hotels=list(getattr(got.get("accommodation"), "options", [])),
                dining=list(getattr(got.get("dining"), "options", [])),
                transport=got.get("transport"),
                weather=getattr(weather, "weather", None),
                events=list(getattr(weather, "events", [])),
            )

    def budget_plan(self) -> BudgetPlan:
        plan = self.accepted.get("budget")
        if not isinstance(plan, BudgetPlan):
            raise RuntimeError("composition needs the budget allocation")
        return plan

    def trip_line(self) -> str:
        """One-line request summary for task descriptions (kickoff input ``trip``)."""
        req = self.requirements.request if self.requirements else None
        if req is None:
            return "the traveller's trip"
        party = f"{req.travellers.adults} adult(s)" + (
            f", {len(req.travellers.children_ages)} child(ren)"
            if req.travellers.children_ages else "")
        interests = ", ".join(req.interests[:5]) or "general sightseeing"
        return (f"{party}, {req.start_date.isoformat()} to {req.end_date.isoformat()}, "
                f"budget AED {req.total_budget_aed()}, interests: {interests}")

    def trace_inputs(self) -> dict[str, Any]:
        """What a stage can read from the board, summarised for the inspector."""
        with self._lock:
            out: dict[str, Any] = {"iteration": self.iteration,
                                   "accepted": sorted(self.accepted)}
            req = self.requirements.request if self.requirements else None
            if req is None:
                out["message"] = self.user_message
            else:
                out["request"] = {"dates": f"{req.start_date} to {req.end_date}",
                                  "travellers": req.travellers.total,
                                  "budget_aed": req.budget_aed, "interests": req.interests}
            if self.feedback:
                out["feedback"] = list(self.feedback)
            if self.locked_days:
                out["locked_days"] = list(self.locked_days)
            return out

    # ------------------------------------------------------------------ hand-over
    def put_pending(self, stage: str, value: BaseModel) -> None:
        with self._lock:
            self._pending[stage] = value

    def take_pending(self, stage: str) -> BaseModel | None:
        with self._lock:
            return self._pending.pop(stage, None)

    def accept(self, stage: str, value: BaseModel) -> None:
        with self._lock:
            self.accepted[stage] = value
            if stage == "requirements" and isinstance(value, requirements.RequirementsResult):
                self.requirements = value

    # ------------------------------------------------------------------ review loop
    def record_attempt(self, attempt: Attempt) -> None:
        with self._lock:
            self.last_attempt = attempt
            self.best = better(attempt, self.best)
            self.attempts.append(attempt_summary(attempt, self.locked_days))

    def start_revision(self, plan: RevisionPlan) -> None:
        """Next compose attempt: one more refinement, with feedback and locked days."""
        with self._lock:
            if self.iteration >= self.max_iterations:
                raise RuntimeError("review-iteration cap reached")
            self.iteration += 1
            self.feedback = list(plan.feedback)
            self.locked_days = list(plan.locked_days)

    @property
    def can_revise(self) -> bool:
        return self.iteration < self.max_iterations

    def feedback_text(self) -> str:
        return " | ".join(self.feedback) if self.feedback else "none (first draft)"

    # ------------------------------------------------------------------ control
    def fail(self, stage: str, exc: BaseException) -> None:
        with self._lock:
            if self.error is None:
                self.error, self.failed_stage = exc, stage

    def stopped_marker(self) -> str | None:
        if self.cancelled:
            return CANCELLED_MARKER
        if self.error is not None:
            return f"{FAILED_MARKER}: {self.failed_stage}"
        return None

    def raise_if_stopped(self) -> None:
        """Called by every guardrail: a recorded failure or a cancel stops the crew."""
        if self.cancelled:
            raise CrewRunCancelled("the planning run was cancelled")
        if self.error is not None:
            raise CrewRunAborted(f"stage {self.failed_stage!r} failed") from self.error


# --------------------------------------------------------------------------- briefs


def _ids(options: list[Any], limit: int = 12) -> str:
    shown = ", ".join(f"{o.id} {o.name}" for o in options[:limit])
    more = f" (+{len(options) - limit} more)" if len(options) > limit else ""
    return shown + more


def brief(stage: str, value: BaseModel) -> str:
    """Short text an agent returns as its answer (and that becomes task context).

    The full typed output stays on the board; keeping the text short keeps every
    downstream prompt small (large JSON contexts would blow free-tier token limits).
    """
    if isinstance(value, requirements.RequirementsResult):
        if value.request is None:
            return f"Request incomplete; missing: {', '.join(value.missing) or 'unknown'}."
        r = value.request
        return (f"Trip request: {r.start_date} to {r.end_date}, "
                f"{r.travellers.adults} adult(s), {len(r.travellers.children_ages)} child(ren), "
                f"budget AED {r.total_budget_aed()}, interests {', '.join(r.interests) or '-'}, "
                f"dietary {', '.join(r.dietary) or '-'}, pace {r.pace.value}.")
    if stage in ("attractions", "accommodation", "dining"):
        options = list(getattr(value, "options", []))
        return f"{len(options)} {stage} option(s) selected: {_ids(options)}."
    if isinstance(value, TransportPlan):
        modes = ", ".join(m.value for m in value.primary_modes)
        return (f"Transport: {modes}; about AED {value.estimated_daily_cost_aed}/day, "
                f"AED {value.estimated_total_cost_aed} in total.")
    if isinstance(value, weather_events.WeatherEventsResult):
        return (f"Weather source {value.weather.source.value}; {len(value.hot_days)} hot day(s); "
                f"{len(value.events)} event(s). " + " ".join(value.advisories[:3]))
    if isinstance(value, BudgetPlan):
        cats = ", ".join(f"{a.category.value} AED {a.amount_aed}" for a in value.allocations)
        state = "feasible" if value.feasible else f"infeasible ({value.infeasibility_reason})"
        return f"Budget AED {value.budget_aed} ({state}): {cats}."
    if isinstance(value, Itinerary):
        return (f"Itinerary: {len(value.days)} day(s), {sum(len(d.slots) for d in value.days)} "
                f"slot(s), total AED {value.cost.total} of AED {value.cost.budget_aed}.")
    if isinstance(value, ReviewReport):
        verdict = "approved" if value.approved else "not approved"
        blocking = "; ".join(i.message for i in value.blocking_issues[:3])
        return (f"Review: {verdict}, score {value.score}."
                + (f" Blocking: {blocking}" if blocking else ""))
    return type(value).__name__
