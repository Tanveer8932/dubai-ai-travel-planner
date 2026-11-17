"""Fault injection for the review loop (implementation plan, Phase 7 task 5; S5, AC7.4).

Each :class:`Defect` corrupts the **first** itinerary the composer produces, the way a
weak model might (over budget, a midday outdoor slot in July, a non-vegetarian
restaurant for a vegetarian…); later compositions are left alone. The corrupted plan
then goes through the mode's real review loop:

- **detected**: the defect's rule appears in the review verdict of that first attempt
  (LangGraph's ``review`` node report; CrewAI's compose-guardrail rejection or review
  report). The verdict merges the deterministic rules with the reviewer agent, so
  with the mock provider detection is by the rules;
- **fixed** (error-severity defects only): the run ends approved and the final plan no
  longer breaks that rule. Warnings do not block approval by design, so they are
  reported as detected, not "fixed".

Injection goes through ``AgentSuite(composer=...)``, the same hook the orchestration
tests use; no framework code is patched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, time
from decimal import Decimal
from pathlib import Path
from typing import Any

from dubai_planner.agents import composer
from dubai_planner.agents.suite import AgentSuite
from dubai_planner.models import (
    ActivityKind,
    AttractionOption,
    BudgetCategory,
    CostBreakdown,
    CostItem,
    DietaryFlag,
    Itinerary,
    PlannerMode,
    PlanStatus,
    Severity,
    TraceEventType,
)
from dubai_planner.models.common import StrictModel
from dubai_planner.services.observability import RunTrace, load_trace
from dubai_planner.services.planner import open_planner
from dubai_planner.storage import Repository
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.validation.rules import validate_itinerary

WINTER = "2 adults, 10-12 Feb 2027, AED 6000, history and food"
SUMMER = "2 adults, 14-16 July 2027, AED 9000, beach and culture"
VEGETARIAN = "2 adults, vegetarian, 10-12 Feb 2027, AED 6000, history and food"
RELAXED = "2 adults, relaxed pace, 10-12 Feb 2027, AED 6000, history and food"
TODAY = date(2026, 11, 16)

Injector = Callable[[Itinerary, "composer.ComposerInput", Toolbox], Itinerary]


@dataclass(frozen=True)
class Defect:
    id: str
    rule_id: str
    severity: Severity
    request: str
    description: str
    inject: Injector


def _slot_index(it: Itinerary, kind: ActivityKind, *, outdoor_ok: bool = True
                ) -> tuple[int, int]:
    for d, day in enumerate(it.days):
        for s, slot in enumerate(day.slots):
            if slot.activity.kind is kind and (outdoor_ok or slot.indoor):
                return d, s
    raise LookupError(f"no {kind.value} slot to corrupt")


def _replace_slot(it: Itinerary, d: int, s: int, **activity: Any) -> Itinerary:
    day = it.days[d]
    slot = day.slots[s]
    new_slot = slot.model_copy(update={"activity": slot.activity.model_copy(update=activity)})
    slots = [*day.slots[:s], new_slot, *day.slots[s + 1:]]
    days = [*it.days[:d], day.model_copy(update={"slots": slots}), *it.days[d + 1:]]
    return it.model_copy(update={"days": days})


def _over_budget(it: Itinerary, data: composer.ComposerInput, _tools: Toolbox) -> Itinerary:
    """A suite upgrade worth the whole budget: costs stay consistent, the total breaks."""
    budget = data.request.total_budget_aed()
    extra = CostItem(category=BudgetCategory.ACCOMMODATION, amount_aed=budget,
                     description="Suite upgrade (injected defect)")
    cost = CostBreakdown.from_items([*it.cost.items, extra], budget_aed=it.cost.budget_aed,
                                    tolerance=it.cost.tolerance)
    return it.model_copy(update={"cost": cost})


def _bad_arithmetic(it: Itinerary, _data: composer.ComposerInput, _tools: Toolbox
                    ) -> Itinerary:
    """Food category raised by AED 100 without changing the total (an LLM sum error)."""
    cost = it.cost.model_copy(update={"food": it.cost.food + Decimal("100")})
    return it.model_copy(update={"cost": cost})


def _midday_outdoor(it: Itinerary, data: composer.ComposerInput, tools: Toolbox) -> Itinerary:
    """Day 1 becomes one outdoor attraction at 12:00-13:30 (assembled like a model reply)."""
    ctx = composer.build_context(data, tools)
    draft = composer.baseline_draft(ctx)
    outdoor = next(a for a in ctx.attractions if not a.indoor and not a.day_trip)
    midday = composer.DraftSlot(start="12:00", end="13:30", kind="attraction", ref_id=outdoor.id)
    first = draft.days[0].model_copy(update={"slots": [midday]})
    return composer.assemble(draft.model_copy(update={"days": [first, *draft.days[1:]]}),
                             data, tools)


def _non_vegetarian(it: Itinerary, _data: composer.ComposerInput, tools: Toolbox) -> Itinerary:
    d, s = _slot_index(it, ActivityKind.DINING)
    meat = next(r for r in tools.catalog.restaurants
                if not r.caters_for([DietaryFlag.VEGETARIAN]))
    return _replace_slot(it, d, s, ref_id=meat.id, name=meat.name, area=meat.area)


def _too_dense(it: Itinerary, _data: composer.ComposerInput, _tools: Toolbox) -> Itinerary:
    """Four extra (free) 12-minute visits squeezed into a free gap of one day: too many
    for a relaxed pace (limit 3 sightseeing slots a day)."""
    def minutes(t: time) -> int:
        return t.hour * 60 + t.minute

    need = 4 * 15 + 5
    for d, day in enumerate(it.days):
        template = next((s for s in day.slots if s.activity.kind is ActivityKind.ATTRACTION),
                        None)
        if template is None:
            continue
        bounds = [(minutes(a.end), minutes(b.start))
                  for a, b in zip(day.slots, day.slots[1:], strict=False)]
        bounds.append((minutes(day.slots[-1].end), 23 * 60 + 59))
        gap = next(((lo, hi) for lo, hi in bounds if hi - lo >= need), None)
        if gap is None:
            continue
        cursor, extra = gap[0] + 5, []
        for _ in range(4):
            extra.append(template.model_copy(update={
                "start": time(cursor // 60, cursor % 60),
                "end": time((cursor + 12) // 60, (cursor + 12) % 60),
                "transport_from_prev": None, "cost_aed": Decimal("0")}))
            cursor += 15
        slots = sorted([*day.slots, *extra], key=lambda s: s.start)
        days = [*it.days[:d], day.model_copy(update={"slots": slots}), *it.days[d + 1:]]
        return it.model_copy(update={"days": days})
    raise LookupError("no day has a free gap for four extra visits")


def _invented_venue(it: Itinerary, _data: composer.ComposerInput, _tools: Toolbox
                    ) -> Itinerary:
    d, s = _slot_index(it, ActivityKind.ATTRACTION)
    return _replace_slot(it, d, s, ref_id="attr_invented_sky_palace",
                         name="Sky Palace of Dubai (does not exist)")


def _long_leg(it: Itinerary, _data: composer.ComposerInput, _tools: Toolbox) -> Itinerary:
    """A 150-minute transfer inside the city (a far-apart pair of venues)."""
    for d, day in enumerate(it.days):
        for s, slot in enumerate(day.slots):
            if slot.transport_from_prev is not None:
                leg = slot.transport_from_prev.model_copy(update={"duration_min": 150})
                new_slot = slot.model_copy(update={"transport_from_prev": leg})
                slots = [*day.slots[:s], new_slot, *day.slots[s + 1:]]
                days = [*it.days[:d], day.model_copy(update={"slots": slots}),
                        *it.days[d + 1:]]
                return it.model_copy(update={"days": days})
    raise LookupError("no transport leg to corrupt")


def _before_opening(it: Itinerary, _data: composer.ComposerInput, tools: Toolbox) -> Itinerary:
    """An extra free visit at 06:00-06:45 to a venue that opens later."""
    day = it.days[0]
    first_start = day.slots[0].start if day.slots else time(23, 0)
    venue: AttractionOption = next(
        a for a in tools.catalog.attractions
        if a.opening_hours is not None and a.opening_hours.open >= time(8, 0)
        and not a.day_trip)
    if first_start <= time(6, 45):
        raise LookupError("day 1 starts too early to prepend a slot")
    template = next(s for s in day.slots if s.activity.kind is ActivityKind.ATTRACTION)
    early = template.model_copy(update={
        "start": time(6, 0), "end": time(6, 45), "transport_from_prev": None,
        "cost_aed": Decimal("0"), "indoor": venue.indoor,
        "activity": template.activity.model_copy(update={
            "ref_id": venue.id, "name": venue.name, "area": venue.area,
            "location": venue.location})})
    days = [day.model_copy(update={"slots": [early, *day.slots]}), *it.days[1:]]
    return it.model_copy(update={"days": days})


DEFECTS: list[Defect] = [
    Defect("over_budget", "budget.total", Severity.ERROR, WINTER,
           "an extra accommodation item worth the whole budget", _over_budget),
    Defect("bad_arithmetic", "budget.arithmetic", Severity.ERROR, WINTER,
           "food category +AED 100 without updating the total", _bad_arithmetic),
    Defect("midday_outdoor_july", "heat.midday_outdoor", Severity.ERROR, SUMMER,
           "an outdoor attraction at 12:00-13:30 in July", _midday_outdoor),
    Defect("non_vegetarian_restaurant", "constraints.dietary", Severity.ERROR, VEGETARIAN,
           "a restaurant without vegetarian options for a vegetarian", _non_vegetarian),
    Defect("before_opening_hours", "time.opening_hours", Severity.WARNING, WINTER,
           "a 06:00 visit to a venue that opens at 08:00 or later", _before_opening),
    Defect("long_transfer", "geo.travel_time", Severity.WARNING, WINTER,
           "a 150-minute transfer between two city venues", _long_leg),
    Defect("invented_venue", "grounding.known_entities", Severity.WARNING, WINTER,
           "an attraction id that is not in the research or seed data", _invented_venue),
    Defect("too_dense_for_pace", "pace.density", Severity.WARNING, RELAXED,
           "four extra evening visits on day 1 of a relaxed-pace trip", _too_dense),
]
# Not injectable: ``constraints.accessibility``. No seed venue is recorded as *not*
# wheelchair accessible (43 attractions yes, 18 unknown; hotels and restaurants all
# unknown), so the rule can only add "access unknown" info notes. Reported as a
# limitation instead of faked with an invented venue.


@dataclass
class InjectingComposer:
    """The real composer; the first itinerary it produces is corrupted by ``defect``."""

    defect: Defect
    calls: int = 0
    injected: Itinerary | None = field(default=None, repr=False)

    def __call__(self, data: composer.ComposerInput, llm: Any, tools: Toolbox) -> Itinerary:
        self.calls += 1
        itinerary = composer.run(data, llm, tools)
        if self.calls == 1:
            itinerary = self.defect.inject(itinerary, data, tools)
            self.injected = itinerary
        return itinerary


class FaultResult(StrictModel):
    defect: str
    rule_id: str
    severity: str
    mode: str
    injected: bool
    present_in_attempt: bool  # the deterministic rules confirm the defect was really there
    detected: bool
    detected_by: list[str]
    fixed: bool | None  # None for warnings (not required to block approval)
    final_status: str | None
    iterations: int = 0
    llm_calls: int | None = None
    error: str | None = None


def findings_at(trace: RunTrace, iteration: int = 0) -> dict[str, set[str]]:
    """Rule ids in the review verdicts recorded for ``iteration`` → which source said so."""
    found: dict[str, set[str]] = {}
    for e in trace.events:
        if e.iteration != iteration:
            continue
        if e.payload.get("output_schema") == "ReviewReport" and "output" in e.payload:
            for issue in e.payload["output"].get("issues", []):
                if issue.get("rule_id"):
                    found.setdefault(issue["rule_id"], set()).add(f"review:{e.agent}")
        if e.event is TraceEventType.GUARDRAIL:
            for row in e.payload.get("issues", []) or []:
                if row.get("rule"):
                    found.setdefault(row["rule"], set()).add(f"guardrail:{e.agent}")
    return found


def run_fault(defect: Defect, mode: PlannerMode, *, repo: Repository, tools: Toolbox,
              checkpoint_path: Path, llm: Any = None, settings: Any = None) -> FaultResult:
    """Inject ``defect`` in one run of ``mode`` and report detection and repair."""
    spy = InjectingComposer(defect)
    kwargs: dict[str, Any] = {"repo": repo, "tools": tools, "agents": AgentSuite(composer=spy)}
    if llm is not None:
        kwargs["llm"] = llm
    if settings is not None:
        kwargs["settings"] = settings
    session = f"fault-{defect.id}-{mode.value}"
    with open_planner(mode, checkpoint_path=checkpoint_path, **kwargs) as planner:
        result = planner.run(defect.request, session, today=TODAY)
    trace = load_trace(repo, result.run_id)
    present = False
    if spy.injected is not None and result.request is not None:
        present = any(v.rule_id == defect.rule_id for v in validate_itinerary(
            spy.injected, result.request, catalog=tools.catalog))
    found = findings_at(trace, 0) if trace is not None else {}
    detected = defect.rule_id in found
    fixed: bool | None = None
    if defect.severity is Severity.ERROR:
        final = result.violations
        fixed = result.status is PlanStatus.APPROVED and not any(
            v.rule_id == defect.rule_id and v.severity is Severity.ERROR for v in final)
    return FaultResult(
        defect=defect.id, rule_id=defect.rule_id, severity=defect.severity.value,
        mode=mode.value, injected=spy.injected is not None, present_in_attempt=present,
        detected=detected, detected_by=sorted(found.get(defect.rule_id, set())), fixed=fixed,
        final_status=result.status.value, iterations=result.iterations,
        llm_calls=trace.run.llm_calls if trace else None, error=result.error)


def detection_rate(results: list[FaultResult], mode: str | None = None) -> float | None:
    own = [r for r in results if r.injected and (mode is None or r.mode == mode)]
    return round(sum(r.detected for r in own) / len(own), 3) if own else None


def fix_rate(results: list[FaultResult], mode: str | None = None) -> float | None:
    own = [r for r in results if r.fixed is not None and (mode is None or r.mode == mode)]
    return round(sum(bool(r.fixed) for r in own) / len(own), 3) if own else None


__all__ = ["DEFECTS", "Defect", "FaultResult", "InjectingComposer",
           "detection_rate", "findings_at", "fix_rate", "run_fault"]
