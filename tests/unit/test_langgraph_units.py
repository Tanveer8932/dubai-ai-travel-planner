"""Phase 3 building blocks: reducers, routers, revision planning, serializer, contracts."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from factories import itinerary, trip_request
from pydantic import BaseModel

from dubai_planner.agents.revision import MAX_FEEDBACK_LINES, plan_revision
from dubai_planner.llm.errors import LLMCallError, StructuredOutputError
from dubai_planner.models import (
    Attempt,
    PlannerMode,
    PlanResult,
    PlanStatus,
    ResearchBundle,
    ReviewIssue,
    ReviewReport,
    RuleViolation,
    Severity,
)
from dubai_planner.orchestration.langgraph import (
    build_graph,
    draw_mermaid,
    is_transient,
    merge_research,
    model_allowlist,
    planner_serializer,
    recursion_limit,
    route_after_parse,
    route_after_review,
)
from dubai_planner.orchestration.langgraph import nodes as n
from dubai_planner.tools.toolbox import Toolbox

ROOT = Path(__file__).parents[2]


def _error(rule: str = "budget.total", day: int | None = None) -> RuleViolation:
    return RuleViolation(rule_id=rule, severity=Severity.ERROR, message=f"{rule} failed",
                         day_index=day)


def _review(approved: bool, *issues: ReviewIssue, score: float = 0.8) -> ReviewReport:
    return ReviewReport(approved=approved, score=score, issues=list(issues))


def _issue(sev: Severity, day: int | None = None, rule: str | None = None) -> ReviewIssue:
    return ReviewIssue(severity=sev, message=f"{sev.value} on {day}", day_index=day,
                       rule_id=rule, suggested_fix="do better")


# ------------------------------------------------------------------ research reducer


def test_merge_research_merges_partial_branch_updates() -> None:
    hotel = itinerary().hotel
    assert hotel is not None
    merged = merge_research(None, {"hotels": [hotel]})
    merged = merge_research(merged, {"events": []})
    assert merged is not None and merged.hotels == [hotel] and merged.attractions == []


def test_merge_research_reset_replace_and_unknown_key() -> None:
    bundle = ResearchBundle()
    assert merge_research(bundle, None) is None
    other = ResearchBundle()
    assert merge_research(bundle, other) is other
    with pytest.raises(ValueError, match="unknown research keys"):
        merge_research(bundle, {"flights": []})


# ------------------------------------------------------------------ routers


def test_route_after_parse() -> None:
    req = trip_request()
    assert route_after_parse({"request": req}, max_clarifications=3) == list(n.RESEARCH_NODES)
    assert route_after_parse({"request": None}, max_clarifications=3) == n.CLARIFY
    assert route_after_parse({"request": None, "clarifications": 3},
                             max_clarifications=3) == n.FINALIZE


def test_route_after_review_never_finalizes_hard_errors_as_approved() -> None:
    ok = _review(True)
    assert route_after_review({"review": ok, "validation": [], "iteration": 0},
                              max_review_iterations=2) == n.FINALIZE
    # The reviewer says approved, but a deterministic rule failed: keep revising.
    assert route_after_review({"review": ok, "validation": [_error()], "iteration": 0},
                              max_review_iterations=2) == n.REFINE
    assert route_after_review({"review": _review(False), "iteration": 1},
                              max_review_iterations=2) == n.REFINE
    assert route_after_review({"review": _review(False), "iteration": 2},
                              max_review_iterations=2) == n.FINALIZE
    assert route_after_review({"review": _review(False), "iteration": 0},
                              max_review_iterations=0) == n.FINALIZE


def test_recursion_limit_is_finite_and_grows_with_caps() -> None:
    assert recursion_limit(2, 3) < recursion_limit(3, 3) < 100


# ------------------------------------------------------------------ revision planning


def test_revision_locks_days_without_blocking_issues() -> None:
    plan = plan_revision(_review(False, _issue(Severity.ERROR, day=1),
                                 _issue(Severity.WARNING, day=0)),
                         itinerary(days=3), num_days=3)
    assert plan.target_days == [1] and plan.locked_days == [0, 2]
    assert plan.feedback[0].startswith("ERROR [reviewer] day 2") and "Fix:" in plan.feedback[0]
    assert plan.feedback[1].startswith("WARNING")


def test_revision_global_error_replans_every_day() -> None:
    plan = plan_revision(_review(False, _issue(Severity.ERROR, rule="budget.total"),
                                 _issue(Severity.ERROR, day=1)), itinerary(days=3), num_days=3)
    assert plan.locked_days == [] and plan.target_days == [0, 1, 2]
    assert "[budget.total]" in plan.feedback[0]


def test_revision_out_of_range_days_and_feedback_bound() -> None:
    plan = plan_revision(_review(False, _issue(Severity.ERROR, day=7)), itinerary(days=2),
                         num_days=2)
    assert plan.locked_days == [] and plan.target_days == [0, 1]
    many = _review(False, *[_issue(Severity.WARNING, day=0) for _ in range(30)])
    assert len(plan_revision(many, itinerary(days=2), num_days=2).feedback) == \
        MAX_FEEDBACK_LINES


# ------------------------------------------------------------------ errors


@pytest.mark.parametrize(("exc", "transient"), [
    (LLMCallError("m: RateLimitError: Error code: 429"), True),
    (LLMCallError("m: APITimeoutError: Request timed out"), True),
    (LLMCallError("m: APIConnectionError: Connection error"), True),
    (LLMCallError("m: InternalServerError: 503 overloaded"), True),
    (LLMCallError("m: AuthenticationError: 401 invalid api key"), False),
    (StructuredOutputError("composer", "DraftItinerary", ["bad"], "x"), False),
    (ValueError("429"), False),
])
def test_is_transient(exc: BaseException, transient: bool) -> None:
    assert is_transient(exc) is transient


# ------------------------------------------------------------------ checkpoint serializer


def test_allowlist_covers_state_models_only() -> None:
    allowed = set(model_allowlist())
    for module, name in [("dubai_planner.models.itinerary", "Itinerary"),
                         ("dubai_planner.models.request", "TripRequest"),
                         ("dubai_planner.models.request", "DietaryFlag"),
                         ("dubai_planner.models.planning", "Attempt"),
                         ("dubai_planner.models.planning", "PlanStatus"),
                         ("dubai_planner.models.review", "Severity"),
                         ("dubai_planner.models.trace", "ChatMessage")]:
        assert (module, name) in allowed
    assert all(m.startswith("dubai_planner.models.") for m, _ in allowed)


def test_strict_serializer_round_trips_state_with_types() -> None:
    it = itinerary()
    attempt = Attempt(iteration=1, itinerary=it, review=_review(True),
                      violations=[_error(day=0)])
    state: dict[str, Any] = {"request": trip_request(), "best": attempt,
                             "status": PlanStatus.NEEDS_REVISION,
                             "research": {"hotels": [it.hotel]}}
    serde = planner_serializer()
    restored = serde.loads_typed(serde.dumps_typed(state))
    assert restored["request"] == state["request"] and restored["best"] == attempt
    assert restored["status"] is PlanStatus.NEEDS_REVISION
    assert restored["research"]["hotels"][0] == it.hotel


class Foreign(BaseModel):
    value: int


def test_strict_serializer_blocks_foreign_classes() -> None:
    serde = planner_serializer()
    restored = serde.loads_typed(serde.dumps_typed({"x": Foreign(value=1)}))
    assert not isinstance(restored["x"], Foreign)  # not in the allowlist: never rebuilt


# ------------------------------------------------------------------ contracts


def test_attempt_approval_needs_reviewer_and_rules() -> None:
    it = itinerary()
    assert Attempt(iteration=0, itinerary=it, review=_review(True)).approved
    assert not Attempt(iteration=0, itinerary=it, review=_review(True),
                       violations=[_error()]).approved
    assert not Attempt(iteration=0, itinerary=it, review=_review(False)).approved


def test_attempt_ranking_prefers_approved_then_fewer_errors() -> None:
    it = itinerary()
    approved = Attempt(iteration=0, itinerary=it, review=_review(True, score=0.5))
    one_error = Attempt(iteration=1, itinerary=it, review=_review(False, score=0.9),
                        violations=[_error()])
    two_errors = Attempt(iteration=2, itinerary=it, review=_review(False, score=0.9),
                         violations=[_error(), _error("time.overlap")])
    llm_error = Attempt(iteration=3, itinerary=it,
                        review=_review(False, _issue(Severity.ERROR, day=0), score=0.9))
    assert approved.rank() > one_error.rank() > two_errors.rank()
    assert llm_error.errors == 1


def test_plan_result_blocking_issues() -> None:
    result = PlanResult(session_id="s", run_id="r", mode=PlannerMode.LANGGRAPH,
                        status=PlanStatus.NEEDS_REVISION, violations=[_error()],
                        review=_review(False, _issue(Severity.ERROR, day=0)))
    assert not result.approved and len(result.blocking_issues) == 2


def test_summary_text_variants() -> None:
    it = itinerary()
    ok = Attempt(iteration=0, itinerary=it, review=_review(True))
    bad = Attempt(iteration=2, itinerary=it, review=_review(False, _issue(Severity.ERROR, 0)))
    assert "Approved" in n.summary_text(PlanStatus.APPROVED, ok, None, 0)
    assert "Not approved" in n.summary_text(PlanStatus.NEEDS_REVISION, bad, None, 2)
    assert n.summary_text(PlanStatus.NEEDS_CLARIFICATION, None, "When?", 0) == "When?"
    assert "could not" in n.summary_text(PlanStatus.NEEDS_REVISION, None, None, 0)
    assert Decimal(str(it.cost.total)) > 0


# ------------------------------------------------------------------ graph shape / mermaid


def test_graph_shape_and_mermaid(toolbox: Toolbox) -> None:
    graph = build_graph(n.PlannerDeps(tools=toolbox))
    nodes = set(graph.get_graph().nodes)
    assert {n.PARSE, n.CLARIFY, *n.RESEARCH_NODES, n.MERGE, n.BUDGET, n.COMPOSE,
            n.VALIDATE, n.REVIEW, n.REFINE, n.FINALIZE} <= nodes
    mermaid = draw_mermaid(graph)
    assert "refine_itinerary" in mermaid and "ask_clarification" in mermaid


def test_exported_mermaid_is_current(toolbox: Toolbox) -> None:
    """docs/generated/langgraph.mmd must match the graph (scripts/export_langgraph.py)."""
    exported = (ROOT / "docs" / "generated" / "langgraph.mmd").read_text(encoding="utf-8")
    assert exported == draw_mermaid(build_graph(n.PlannerDeps(tools=toolbox)))
