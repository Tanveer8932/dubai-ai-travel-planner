"""Evaluation metrics (implementation plan, Phase 7 task 2; ``research.md §5``).

Every number here is computed by code from what a run *produced and recorded*:

- **quality** from the final :class:`Itinerary`, re-checked independently of the
  planner: the deterministic rules are run again, and each hard constraint is checked
  against the scenario's expectations, which come from the user's words (so a parsing
  mistake counts against the mode);
- **performance** from the ``runs`` rows and their trace events in SQLite (duration,
  LLM calls, tokens, review iterations, research overlap). Nothing is estimated: a
  value that was not recorded stays ``None``.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import time
from decimal import Decimal
from typing import Any

from pydantic import Field

from dubai_planner.data.seed import SeedCatalog
from dubai_planner.models import (
    ActivityKind,
    BudgetPlan,
    DiningOption,
    Itinerary,
    PlanResult,
    PlanStatus,
    Severity,
    TraceEvent,
    TraceEventType,
)
from dubai_planner.models.common import StrictModel
from dubai_planner.models.request import AccessibilityNeed
from dubai_planner.redaction import redact
from dubai_planner.services.observability import RunTrace, parallelism, step_spans
from dubai_planner.tools.search import interest_matches, venue_text
from dubai_planner.validation.rules import validate_itinerary
from dubai_planner.validation.thresholds import (
    MIDDAY_END,
    MIDDAY_START,
    PACE_MAX_ACTIVITIES,
)

from .scenarios import ExpectedOutcome, Scenario

SIGHTSEEING = frozenset({ActivityKind.ATTRACTION, ActivityKind.EVENT})
_ACCOUNT_IDS = re.compile(r"\b(org|user|proj|acct)_[A-Za-z0-9]{6,}")


def scrub(text: str | None) -> str | None:
    """Errors kept in results: configured keys and provider account ids masked.

    Provider rate-limit messages name the account's organisation id; results are
    committed and published, so those ids are replaced.
    """
    if text is None:
        return None
    return _ACCOUNT_IDS.sub(lambda m: f"{m.group(1)}_***", redact(text))


@dataclass
class TurnOutcome:
    """One user turn of a scenario as executed (what the runner hands to the scorer)."""

    message: str
    routing: str  # services.refinement ChangeKind
    explanation: str
    result: PlanResult | None
    trace: RunTrace | None
    error: str | None = None
    wall_s: float | None = None


class TurnRecord(StrictModel):
    """Per-turn facts kept in the results (input, routing, outcome, cost)."""

    message: str
    routing: str
    explanation: str = ""
    run_id: str | None = None
    status: str | None = None
    saved_version: int | None = None
    duration_s: float | None = None
    llm_calls: int | None = None
    tokens: int | None = None
    iterations: int | None = None
    total_aed: float | None = None
    error: str | None = None


class ConstraintCheck(StrictModel):
    name: str
    satisfied: bool
    detail: str = ""


class CellMetrics(StrictModel):
    """Metrics of one scenario × mode × seed cell."""

    scenario_id: str
    category: str
    mode: str
    seed: int = 0
    provider: str = ""
    model: str = ""
    # outcome
    final_status: str | None = None
    expected_outcome: str = ExpectedOutcome.APPROVED.value
    outcome_ok: bool = False
    failed: bool = False
    error: str | None = None
    # validity (S1)
    itinerary_expected: bool = True
    schema_valid: bool | None = None
    # budget (S2)
    budget_aed: float | None = None
    total_aed: float | None = None
    overshoot_pct: float | None = None
    within_tolerance: bool | None = None
    infeasibility_explained: bool | None = None
    budget_compliant: bool | None = None
    # hard constraints (S3)
    constraints: list[ConstraintCheck] = Field(default_factory=list)
    constraints_checked: int = 0
    constraints_satisfied: int = 0
    constraint_rate: float | None = None
    # rule re-check of the final plan
    violations_error: int | None = None
    violations_warning: int | None = None
    violations_info: int | None = None
    violation_rules: list[str] = Field(default_factory=list)
    # feasibility / preferences
    mean_leg_km: float | None = None
    mean_daily_km: float | None = None
    max_leg_min: int | None = None
    grounding_rate: float | None = None
    interest_match: float | None = None
    indoor_share: float | None = None
    # performance (summed over turns, from the run rows)
    latency_s: float | None = None
    llm_calls: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens: int | None = None
    review_iterations: int | None = None
    guardrail_rejections: int | None = None
    delegations: int | None = None
    research_window_s: float | None = None
    research_sum_s: float | None = None
    research_ratio: float | None = Field(
        default=None, description="research window / sum of research step durations (S4)")
    # optional judge
    judge_score: float | None = None
    judge_kind: str | None = None
    turns: list[TurnRecord] = Field(default_factory=list)


#: Numeric metrics aggregated as mean ± std in reports (column, label, unit).
NUMERIC_METRICS: list[tuple[str, str]] = [
    ("schema_valid", "Valid itinerary (share)"),
    ("budget_compliant", "Budget compliant (share)"),
    ("outcome_ok", "Expected outcome (share)"),
    ("constraint_rate", "Hard constraints satisfied (share)"),
    ("overshoot_pct", "Total vs budget (%)"),
    ("violations_error", "Rule errors"),
    ("violations_warning", "Rule warnings"),
    ("interest_match", "Interest match (share)"),
    ("grounding_rate", "Grounded venues (share)"),
    ("mean_daily_km", "Travel per day (km)"),
    ("latency_s", "Latency (s)"),
    ("llm_calls", "LLM calls"),
    ("tokens", "Tokens"),
    ("review_iterations", "Review iterations"),
    ("guardrail_rejections", "Guardrail rejections"),
    ("research_ratio", "Research window / sum"),
    ("judge_score", "Judge score (1-5)"),
]


# --------------------------------------------------------------------------- helpers


def _venue(catalog: SeedCatalog, ref: str | None) -> Any:
    return catalog.venue(ref) if ref else None


def _overlaps_midday(start: time, end: time) -> bool:
    return start < MIDDAY_END and end > MIDDAY_START


def budget_plan_from(trace: RunTrace | None) -> BudgetPlan | None:
    """The run's recorded :class:`BudgetPlan` (the budget step's validated output)."""
    if trace is None:
        return None
    for event in reversed(trace.events):
        if event.payload.get("output_schema") == "BudgetPlan" and "output" in event.payload:
            try:
                return BudgetPlan.model_validate(event.payload["output"])
            except ValueError:
                return None
    return None


def check_constraints(scenario: Scenario, result: PlanResult, catalog: SeedCatalog
                      ) -> list[ConstraintCheck]:
    """Hard constraints of the scenario, checked on the final itinerary."""
    exp, it, req = scenario.expect, result.itinerary, result.request
    if it is None:
        return []
    out: list[ConstraintCheck] = []
    slots = [(d, s) for d, day in enumerate(it.days) for s in day.slots]

    def add(name: str, ok: bool, detail: str = "") -> None:
        out.append(ConstraintCheck(name=name, satisfied=ok, detail=detail))

    if exp.days is not None:
        add("days", len(it.days) == exp.days, f"{len(it.days)} planned, {exp.days} asked")
    if exp.adults is not None or exp.children is not None:
        got = (req.travellers.adults, req.travellers.children) if req else (None, None)
        ok = req is not None and (exp.adults is None or got[0] == exp.adults) and (
            exp.children is None or got[1] == exp.children)
        add("travellers", ok, f"parsed adults={got[0]} children={got[1]}")
    if exp.dietary:
        bad = []
        for _d, slot in slots:
            if slot.activity.kind is ActivityKind.DINING:
                venue = _venue(catalog, slot.activity.ref_id)
                if not isinstance(venue, DiningOption) or not venue.caters_for(exp.dietary):
                    bad.append(slot.activity.name)
        add("dietary", not bad, ", ".join(bad[:5]) or "all restaurants cater for it")
    if AccessibilityNeed.WHEELCHAIR in exp.accessibility:
        refs = [s.activity.ref_id for _d, s in slots] + ([it.hotel.id] if it.hotel else [])
        bad = sorted({v.name for v in (_venue(catalog, r) for r in refs)
                      if v is not None and v.wheelchair_accessible is False})
        add("accessibility", not bad, ", ".join(bad) or "no venue recorded as inaccessible")
    if exp.pace is not None:
        limit = PACE_MAX_ACTIVITIES[exp.pace]
        counts = [sum(1 for s in day.slots if s.activity.kind in SIGHTSEEING) for day in it.days]
        add("pace", max(counts, default=0) <= limit, f"max {max(counts, default=0)}/day, "
            f"limit {limit} ({exp.pace.value})")
    if exp.heat_safe:
        bad = [f"day {d + 1} {s.start:%H:%M} {s.activity.name}" for d, s in slots
               if not s.indoor and s.activity.kind not in (ActivityKind.TRANSIT,
                                                           ActivityKind.HOTEL)
               and _overlaps_midday(s.start, s.end)]
        add("heat_safe", not bad, "; ".join(bad[:3]) or "no outdoor slot 11:00-16:00")
    if exp.family_friendly:
        bad = sorted({v.name for _d, s in slots if s.activity.kind is ActivityKind.ATTRACTION
                      for v in [_venue(catalog, s.activity.ref_id)]
                      if v is not None and not v.family_friendly})
        add("family_friendly", not bad, ", ".join(bad) or "all attractions family-friendly")
    if exp.latest_end is not None:
        last = max((s.end for _d, s in slots), default=None)
        add("latest_end", last is None or last <= exp.latest_end,
            f"last slot ends {last:%H:%M}" if last else "no slots")
    if exp.day_trip_area:
        hit = any(getattr(_venue(catalog, s.activity.ref_id), "area", None) == exp.day_trip_area
                  for _d, s in slots if s.activity.kind is ActivityKind.ATTRACTION)
        add("day_trip", hit, f"{'an' if hit else 'no'} attraction in {exp.day_trip_area}")
    for day in exp.indoor_days:
        idx = day - 1
        attractions = [s for s in it.days[idx].slots
                       if s.activity.kind is ActivityKind.ATTRACTION] if idx < len(it.days) else []
        add(f"indoor_day_{day}", bool(attractions) and all(s.indoor for s in attractions),
            f"{sum(s.indoor for s in attractions)}/{len(attractions)} attractions indoor")
    return out


def _geo(it: Itinerary) -> tuple[float | None, float | None, int | None]:
    legs = [s.transport_from_prev for d in it.days for s in d.slots if s.transport_from_prev]
    if not legs:
        return None, None, None
    per_day = [sum(s.transport_from_prev.distance_km for s in d.slots if s.transport_from_prev)
               for d in it.days]
    return (round(statistics.fmean(leg.distance_km for leg in legs), 2),
            round(statistics.fmean(per_day), 2), max(leg.duration_min for leg in legs))


def _interest_match(it: Itinerary, interests: list[str], catalog: SeedCatalog) -> float | None:
    if not interests:
        return None
    texts = [venue_text(v) for d in it.days for s in d.slots
             if s.activity.kind in (ActivityKind.ATTRACTION, ActivityKind.DINING)
             for v in [_venue(catalog, s.activity.ref_id)] if v is not None]
    matched = sum(1 for i in interests if any(interest_matches(t, [i]) for t in texts))
    return round(matched / len(interests), 3)


def _explained(result: PlanResult, plan: BudgetPlan | None) -> bool:
    """The over-budget plan says so (budget step's reason or a note on the itinerary)."""
    if plan is not None and not plan.feasible and plan.infeasibility_reason:
        return True
    notes = result.itinerary.notes if result.itinerary else []
    return any("exceeds the budget" in n.lower() for n in notes)


def _sum(values: Iterable[float | int | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals), 3) if vals else None


def _count(events: Sequence[TraceEvent], kind: TraceEventType, **match: Any) -> int:
    return sum(1 for e in events if e.event is kind
               and all(e.payload.get(k) == v for k, v in match.items()))


# --------------------------------------------------------------------------- scoring


def score(scenario: Scenario, turns: Sequence[TurnOutcome], *, mode: str, seed: int,
          catalog: SeedCatalog, tolerance: Decimal, provider: str = "", model: str = ""
          ) -> CellMetrics:
    """All metrics of one executed scenario (the final turn's plan, all turns' cost)."""
    exp = scenario.expect
    last = turns[-1] if turns else None
    result = last.result if last else None
    m = CellMetrics(scenario_id=scenario.id, category=scenario.category, mode=mode, seed=seed,
                    provider=provider, model=model, expected_outcome=exp.outcome.value,
                    itinerary_expected=exp.outcome is not ExpectedOutcome.NEEDS_CLARIFICATION)
    m.turns = [_turn_record(t) for t in turns]
    errors = [t.error or (t.result.error if t.result else None) for t in turns]
    m.error = scrub(next((e for e in errors if e), None))
    m.failed = result is None or result.status is PlanStatus.FAILED or any(
        t.result is None or t.result.status is PlanStatus.FAILED for t in turns)
    m.final_status = result.status.value if result else None
    _performance(m, turns)
    if result is None:
        m.schema_valid = False if m.itinerary_expected else None
        return m
    it = result.itinerary
    if m.itinerary_expected:
        m.schema_valid = it is not None and _round_trips(it)
    first = turns[0].result
    asked = first is not None and first.status is PlanStatus.NEEDS_CLARIFICATION
    if exp.outcome is ExpectedOutcome.NEEDS_CLARIFICATION:
        m.outcome_ok = result.status is PlanStatus.NEEDS_CLARIFICATION
        return m
    if it is None:
        return m
    budget = Decimal(str(exp.budget_aed)) if exp.budget_aed is not None else (
        result.request.total_budget_aed() if result.request else None)
    m.total_aed = float(it.cost.total)
    if budget is not None:
        m.budget_aed = float(budget)
        m.overshoot_pct = round(float((it.cost.total - budget) / budget * 100), 2)
        m.within_tolerance = it.cost.total <= budget * (1 + tolerance)
        m.infeasibility_explained = (not m.within_tolerance) and _explained(
            result, budget_plan_from(last.trace if last else None))
        m.budget_compliant = bool(m.within_tolerance or m.infeasibility_explained)
    m.constraints = check_constraints(scenario, result, catalog)
    m.constraints_checked = len(m.constraints)
    m.constraints_satisfied = sum(c.satisfied for c in m.constraints)
    m.constraint_rate = (round(m.constraints_satisfied / m.constraints_checked, 3)
                         if m.constraints_checked else None)
    if result.request is not None:
        found = validate_itinerary(it, result.request, catalog=catalog, tolerance=tolerance)
        m.violations_error = sum(v.severity is Severity.ERROR for v in found)
        m.violations_warning = sum(v.severity is Severity.WARNING for v in found)
        m.violations_info = sum(v.severity is Severity.INFO for v in found)
        m.violation_rules = sorted({v.rule_id for v in found if v.severity is not Severity.INFO})
    m.mean_leg_km, m.mean_daily_km, m.max_leg_min = _geo(it)
    ids = it.venue_ids()
    known = catalog.known_ids()
    m.grounding_rate = round(len(ids & known) / len(ids), 3) if ids else None
    m.interest_match = _interest_match(it, exp.interests, catalog)
    attractions = [s for d in it.days for s in d.slots
                   if s.activity.kind is ActivityKind.ATTRACTION]
    m.indoor_share = (round(sum(s.indoor for s in attractions) / len(attractions), 3)
                      if attractions else None)
    status_ok = (result.status is PlanStatus.APPROVED and bool(m.within_tolerance)
                 if exp.outcome is ExpectedOutcome.APPROVED
                 else bool(m.infeasibility_explained))
    m.outcome_ok = status_ok and (asked or not exp.asks_clarification)
    return m


def _round_trips(it: Itinerary) -> bool:
    try:
        return Itinerary.model_validate_json(it.model_dump_json()) == it
    except ValueError:
        return False


def _turn_record(t: TurnOutcome) -> TurnRecord:
    run = t.trace.run if t.trace else None
    r = t.result
    return TurnRecord(
        message=t.message, routing=t.routing, explanation=t.explanation,
        run_id=r.run_id if r else None, status=r.status.value if r else None,
        saved_version=r.saved_version if r else None,
        duration_s=run.duration_s if run and run.duration_s is not None else t.wall_s,
        llm_calls=run.llm_calls if run else None, tokens=run.tokens if run else None,
        iterations=r.iterations if r else None,
        total_aed=float(r.itinerary.cost.total) if r and r.itinerary else None,
        error=scrub(t.error or (r.error if r else None)))


def _performance(m: CellMetrics, turns: Sequence[TurnOutcome]) -> None:
    traces = [t.trace for t in turns if t.trace is not None]
    m.latency_s = _sum(t.trace.run.duration_s if t.trace and t.trace.run.duration_s is not None
                       else t.wall_s for t in turns)
    if not traces:
        return
    m.llm_calls = sum(tr.run.llm_calls for tr in traces)
    m.tokens_in = sum(tr.run.tokens_in for tr in traces)
    m.tokens_out = sum(tr.run.tokens_out for tr in traces)
    m.tokens = sum(tr.run.tokens for tr in traces)
    m.review_iterations = sum((t.result.iterations if t.result else 0) for t in turns)
    events = [e for tr in traces for e in tr.events]
    m.guardrail_rejections = _count(events, TraceEventType.GUARDRAIL, passed=False)
    m.delegations = _count(events, TraceEventType.DELEGATION)
    for tr in traces:  # the first turn that ran the research fan-out
        par = parallelism(step_spans(tr))
        if par is not None and par.steps >= 2 and par.sum_s > 0:
            m.research_window_s, m.research_sum_s = par.window_s, par.sum_s
            m.research_ratio = round(par.window_s / par.sum_s, 3)
            break


# --------------------------------------------------------------------------- aggregation


@dataclass(frozen=True)
class Stat:
    n: int
    mean: float | None
    std: float | None

    def show(self, digits: int = 2) -> str:
        if self.mean is None:
            return "—"
        std = f" ± {self.std:.{digits}f}" if self.std is not None else ""
        return f"{self.mean:.{digits}f}{std} (n={self.n})"


def stat(values: Iterable[Any]) -> Stat:
    """Mean ± sample std of the recorded values (booleans count as 0/1; None skipped)."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return Stat(0, None, None)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return Stat(len(vals), statistics.fmean(vals), std)


def aggregate(cells: Sequence[CellMetrics]) -> dict[str, dict[str, Stat]]:
    """``{mode: {metric: Stat}}`` over every cell of that mode."""
    out: dict[str, dict[str, Stat]] = {}
    for mode in dict.fromkeys(c.mode for c in cells):
        own = [c for c in cells if c.mode == mode]
        out[mode] = {key: stat(getattr(c, key) for c in own) for key, _ in NUMERIC_METRICS}
    return out
