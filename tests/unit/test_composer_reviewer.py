"""Composer (draft → grounded, priced Itinerary) and Reviewer (rules + LLM, rule-decided)."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from factories import compose, research_bundle, trip_request

from dubai_planner.agents import budget, composer, reviewer
from dubai_planner.agents.composer import (
    ComposerInput,
    DraftDay,
    DraftItinerary,
    DraftSlot,
    assemble,
    baseline_draft,
    build_context,
    parse_time,
)
from dubai_planner.llm import StructuredOutputError
from dubai_planner.llm.mock import MockLLM
from dubai_planner.models import (
    ActivityKind,
    BudgetPlan,
    PlannerMode,
    ResearchBundle,
    RuleViolation,
    Severity,
    TransportMode,
    TripRequest,
)
from dubai_planner.models.trace import TraceEventType
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import capture
from dubai_planner.validation.rules import validate_itinerary


@pytest.fixture
def req() -> TripRequest:
    return trip_request(budget_aed=Decimal("15000"))


@pytest.fixture
def research(req: TripRequest, mock_llm: MockLLM, toolbox: Toolbox) -> ResearchBundle:
    return research_bundle(req, mock_llm, toolbox)


@pytest.fixture
def plan(req: TripRequest, research: ResearchBundle, mock_llm: MockLLM, toolbox: Toolbox
         ) -> BudgetPlan:
    return budget.run(budget.BudgetInput(request=req, research=research), mock_llm, toolbox)


def _input(req: TripRequest, research: ResearchBundle, plan: BudgetPlan, **kw: Any
           ) -> ComposerInput:
    return ComposerInput(session_id="s1", mode=PlannerMode.LANGGRAPH, request=req,
                         research=research, budget=plan, **kw)


def _slot(start: str, end: str, kind: str, ref: str | None = None, **kw: Any) -> DraftSlot:
    return DraftSlot(start=start, end=end, kind=kind, ref_id=ref, **kw)  # type: ignore[arg-type]


def _draft(research: ResearchBundle, n_days: int, **overrides: Any) -> DraftItinerary:
    att = [a.id for a in research.attractions]
    din = [d.id for d in research.dining]
    days = [DraftDay(day=i + 1, slots=[
        _slot("09:00", "10:00", "attraction", att[i]),
        _slot("12:30", "13:30", "dining", din[i % len(din)]),
    ]) for i in range(n_days)]
    data: dict[str, Any] = {"title": "T", "hotel_id": research.hotels[0].id, "days": days}
    data.update(overrides)
    return DraftItinerary(**data)


# ------------------------------------------------------------------ assembly


@pytest.mark.parametrize(("raw", "ok"), [("09:30", True), ("9:05", True), ("23:59:00", True),
                                        ("24:00", False), ("12:60", False), ("noon", False)])
def test_parse_time(raw: str, ok: bool) -> None:
    if ok:
        parse_time(raw, "t")
    else:
        with pytest.raises(ValueError, match="valid HH:MM"):
            parse_time(raw, "t")


def test_assemble_grounds_prices_and_routes(req: TripRequest, research: ResearchBundle,
                                            plan: BudgetPlan, toolbox: Toolbox) -> None:
    it = assemble(_draft(research, req.num_days), _input(req, research, plan), toolbox)
    first = it.days[0].slots[0]
    venue = next(a for a in research.attractions if a.id == first.activity.ref_id)
    assert first.activity.name == venue.name and first.indoor is venue.indoor
    assert first.activity.location == venue.location
    assert first.transport_from_prev is not None
    assert first.transport_from_prev.from_ref == research.hotels[0].id
    assert it.cost.is_consistent and it.cost.budget_aed == req.total_budget_aed()
    assert it.hotel == research.hotels[0] and it.notes[0].startswith("All prices are estimates")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.days.pop(), "missing day"),
        (lambda d: d.days.append(DraftDay(day=99)), "outside the trip"),
        (lambda d: d.days.append(d.days[0]), "repeated"),
        (lambda d: d.days[0].slots.append(_slot("09:30", "10:30", "free_time", title="x")),
         "overlaps"),
        (lambda d: d.days[0].slots.append(_slot("15:00", "14:00", "free_time", title="x")),
         "must be after"),
        (lambda d: d.days[0].slots.append(_slot("7 am", "08:00", "free_time", title="x")),
         "valid HH:MM"),
        (lambda d: d.days[0].slots.append(_slot("20:00", "21:00", "dining", "res_nowhere")),
         "not in the provided dining list"),
        (lambda d: d.days[0].slots.append(_slot("20:00", "21:00", "event", "evt_nowhere")),
         "not a listed event"),
        (lambda d: d.days[1].slots.append(_slot("16:00", "17:00", "attraction",
                                                d.days[0].slots[0].ref_id)), "more than once"),
    ],
)
def test_assemble_reports_problems(req: TripRequest, research: ResearchBundle, plan: BudgetPlan,
                                   toolbox: Toolbox, mutate: Any, message: str) -> None:
    draft = _draft(research, req.num_days)
    mutate(draft)
    with pytest.raises(ValueError, match=message):
        assemble(draft, _input(req, research, plan), toolbox)


def test_unknown_hotel_reported(req: TripRequest, research: ResearchBundle, plan: BudgetPlan,
                                toolbox: Toolbox) -> None:
    with pytest.raises(ValueError, match="hotel_id"):
        assemble(_draft(research, req.num_days, hotel_id="htl_nowhere"),
                 _input(req, research, plan), toolbox)


def test_missing_hotel_defaults_to_recommended(req: TripRequest, research: ResearchBundle,
                                               plan: BudgetPlan, toolbox: Toolbox) -> None:
    it = assemble(_draft(research, req.num_days, hotel_id=None), _input(req, research, plan),
                  toolbox)
    assert it.hotel == research.hotels[0]


def test_free_time_and_events(req: TripRequest, research: ResearchBundle, plan: BudgetPlan,
                              toolbox: Toolbox) -> None:
    draft = _draft(research, req.num_days)
    draft.days[0].slots.append(_slot("15:00", "16:00", "free_time", title="Pool"))
    if research.events:
        draft.days[0].slots.append(_slot("18:00", "19:00", "event", research.events[0].id))
    it = assemble(draft, _input(req, research, plan), toolbox)
    kinds = [s.activity.kind for s in it.days[0].slots]
    assert ActivityKind.FREE_TIME in kinds
    free = next(s for s in it.days[0].slots if s.activity.kind is ActivityKind.FREE_TIME)
    assert free.activity.name == "Pool" and free.cost_aed == 0


def test_tour_pickup_for_desert_safari(toolbox: Toolbox, mock_llm: MockLLM) -> None:
    req = trip_request(interests=["desert"], travellers=trip_request().travellers,
                       budget_aed=Decimal("30000"))
    research = research_bundle(req, mock_llm, toolbox)
    safari = next((a for a in research.attractions if "tour pickup" in a.tags), None)
    if safari is None:
        pytest.skip("no tour-pickup attraction selected")
    plan = budget.run(budget.BudgetInput(request=req, research=research), mock_llm, toolbox)
    draft = DraftItinerary(hotel_id=research.hotels[0].id, days=[
        DraftDay(day=d, slots=[_slot("17:00", "21:00", "attraction", safari.id)] if d == 1
                 else []) for d in range(1, req.num_days + 1)])
    it = assemble(draft, _input(req, research, plan), toolbox)
    leg = it.days[0].slots[0].transport_from_prev
    assert leg is not None and leg.mode is TransportMode.TOUR_PICKUP and leg.cost_aed == 0


def test_locked_days_are_copied_from_previous(req: TripRequest, research: ResearchBundle,
                                              plan: BudgetPlan, mock_llm: MockLLM,
                                              toolbox: Toolbox) -> None:
    previous = composer.run(_input(req, research, plan), mock_llm, toolbox)
    revised = composer.run(_input(req, research, plan, previous=previous, locked_days=[0, 2],
                                  feedback=["Make day 2 indoor"]), mock_llm, toolbox)
    assert revised.days[0] == previous.days[0] and revised.days[2] == previous.days[2]
    sent = mock_llm.calls[-1].messages[-1].content
    assert "Make day 2 indoor" in sent and '"locked_days": [1, 3]' in sent


def test_locked_days_need_a_valid_previous(req: TripRequest, research: ResearchBundle,
                                           plan: BudgetPlan, mock_llm: MockLLM,
                                           toolbox: Toolbox) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="requires the previous"):
        _input(req, research, plan, locked_days=[0])
    previous = composer.run(_input(req, research, plan), mock_llm, toolbox)
    with pytest.raises(ValidationError, match="outside the previous"):
        _input(req, research, plan, previous=previous, locked_days=[7])


# ------------------------------------------------------------------ baseline scheduler


def test_baseline_respects_heat_pace_and_budget(toolbox: Toolbox, mock_llm: MockLLM) -> None:
    req = trip_request(start_date=date(2027, 7, 12), end_date=date(2027, 7, 15),
                       budget_aed=Decimal("12000"), interests=["desert", "views", "beach"])
    research, _plan, it = compose(req, mock_llm, toolbox)
    violations = validate_itinerary(it, req, research, catalog=toolbox.catalog)
    assert not [v for v in violations if v.rule_id == "heat.midday_outdoor"]
    assert not [v for v in violations if v.rule_id in ("pace.density", "time.overlap")]
    assert it.cost.within_budget


def test_baseline_draft_is_valid_for_assembly(req: TripRequest, research: ResearchBundle,
                                              plan: BudgetPlan, toolbox: Toolbox) -> None:
    data = _input(req, research, plan)
    draft = baseline_draft(build_context(data, toolbox))
    assert [d.day for d in draft.days] == list(range(1, req.num_days + 1))
    assemble(draft, data, toolbox)  # does not raise


# ------------------------------------------------------------------ composer run


def test_composer_repairs_malformed_draft(req: TripRequest, research: ResearchBundle,
                                          plan: BudgetPlan, mock_llm: MockLLM,
                                          toolbox: Toolbox) -> None:
    mock_llm.script("composer", "```json\n{\"days\": [}\n```")
    it = composer.run(_input(req, research, plan), mock_llm, toolbox)
    assert len(it.days) == req.num_days


def test_composer_fails_loudly_without_fallback(req: TripRequest, research: ResearchBundle,
                                                plan: BudgetPlan, mock_llm: MockLLM,
                                                toolbox: Toolbox) -> None:
    mock_llm.script("composer", "{}", "{}")
    with pytest.raises(StructuredOutputError):
        composer.run(_input(req, research, plan), mock_llm, toolbox)


def test_composer_fallback_when_enabled(req: TripRequest, research: ResearchBundle,
                                        plan: BudgetPlan, mock_llm: MockLLM,
                                        toolbox: Toolbox) -> None:
    mock_llm.script("composer", "{}", "{}")
    with capture() as events:
        it = composer.run(_input(req, research, plan), mock_llm, toolbox, fallback=True)
    assert len(it.days) == req.num_days and "deterministic planner" in it.notes[-1]
    assert any(e.event is TraceEventType.INFO and e.payload.get("fallback") == "baseline_draft"
               for e in events)


def test_infeasible_budget_noted(toolbox: Toolbox, mock_llm: MockLLM) -> None:
    req = trip_request(budget_aed=Decimal("900"))
    _r, plan, it = compose(req, mock_llm, toolbox)
    assert not plan.feasible and plan.infeasibility_reason in it.notes


# ------------------------------------------------------------------ reviewer


def _violation(severity: Severity, rule: str = "budget.total") -> RuleViolation:
    return RuleViolation(rule_id=rule, severity=severity, message=f"{rule} {severity.value}")


def _review(req: TripRequest, research: ResearchBundle, plan: BudgetPlan, mock_llm: MockLLM,
            toolbox: Toolbox, violations: list[RuleViolation], **kw: Any) -> Any:
    it = composer.run(_input(req, research, plan), mock_llm, toolbox)
    return reviewer.run(reviewer.ReviewerInput(itinerary=it, request=req,
                                               violations=violations), mock_llm, **kw)


def test_rule_error_blocks_approval(req: TripRequest, research: ResearchBundle, plan: BudgetPlan,
                                   mock_llm: MockLLM, toolbox: Toolbox) -> None:
    mock_llm.script("reviewer", json.dumps({"approved": True, "score": 0.95}))
    report = _review(req, research, plan, mock_llm, toolbox, [_violation(Severity.ERROR)])
    assert not report.approved and report.score <= 0.75
    issue = report.issues[0]
    assert issue.rule_id == "budget.total" and issue.suggested_fix
    assert "rejected by rule" in report.summary


def test_llm_error_issue_blocks_approval(req: TripRequest, research: ResearchBundle,
                                         plan: BudgetPlan, mock_llm: MockLLM, toolbox: Toolbox
                                         ) -> None:
    mock_llm.script("reviewer", json.dumps({"approved": True, "score": 0.9, "issues": [
        {"severity": "error", "message": "No must-see Burj Khalifa", "day": 2}]}))
    report = _review(req, research, plan, mock_llm, toolbox, [])
    assert not report.approved and report.issues[0].day_index == 1


def test_warnings_only_are_approved_by_rule(req: TripRequest, research: ResearchBundle,
                                            plan: BudgetPlan, mock_llm: MockLLM,
                                            toolbox: Toolbox) -> None:
    mock_llm.script("reviewer", json.dumps({"approved": False, "score": 0.6, "summary": "Meh.",
                                            "issues": [{"severity": "warning",
                                                        "message": "Day 1 is long"}]}))
    report = _review(req, research, plan, mock_llm, toolbox,
                     [_violation(Severity.WARNING, "pace.density")])
    assert report.approved and report.score == 0.6
    assert "approved by rule" in report.summary and len(report.blocking_issues) == 0


def test_duplicate_llm_issue_merged(req: TripRequest, research: ResearchBundle, plan: BudgetPlan,
                                    mock_llm: MockLLM, toolbox: Toolbox) -> None:
    v = _violation(Severity.WARNING, "pace.density")
    mock_llm.script("reviewer", json.dumps({"approved": True, "score": 0.8, "issues": [
        {"severity": "warning", "message": v.message}]}))
    report = _review(req, research, plan, mock_llm, toolbox, [v])
    assert len(report.issues) == 1


def test_reviewer_malformed_then_fallback(req: TripRequest, research: ResearchBundle,
                                          plan: BudgetPlan, mock_llm: MockLLM,
                                          toolbox: Toolbox) -> None:
    mock_llm.script("reviewer", "LGTM!", '{"approved": "maybe"}')
    with pytest.raises(StructuredOutputError):
        _review(req, research, plan, mock_llm, toolbox, [])
    mock_llm.script("reviewer", "LGTM!", '{"approved": "maybe"}')
    report = _review(req, research, plan, mock_llm, toolbox, [_violation(Severity.ERROR)],
                     fallback=True)
    assert not report.approved and "Rule check" in report.summary


def test_iteration_recorded(req: TripRequest, research: ResearchBundle, plan: BudgetPlan,
                            mock_llm: MockLLM, toolbox: Toolbox) -> None:
    it = composer.run(_input(req, research, plan), mock_llm, toolbox)
    report = reviewer.run(reviewer.ReviewerInput(itinerary=it, request=req, iteration=2),
                          mock_llm)
    assert report.iteration == 2 and report.approved
