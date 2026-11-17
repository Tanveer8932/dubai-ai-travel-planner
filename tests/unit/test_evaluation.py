"""Phase 7 evaluation package: scenarios, metrics, judge, report, results loader."""

from __future__ import annotations

import sys
from datetime import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from factories import itinerary, slot, trip_request

from dubai_planner.data.seed import load_catalog
from dubai_planner.evaluation import judge
from dubai_planner.evaluation.faults import FaultResult, detection_rate, fix_rate
from dubai_planner.evaluation.metrics import (
    CellMetrics,
    TurnOutcome,
    aggregate,
    check_constraints,
    score,
    stat,
)
from dubai_planner.evaluation.report import MOCK_BANNER, write_report
from dubai_planner.evaluation.results import load_database, load_published
from dubai_planner.evaluation.runner import EvalConfig, EvalOutcome, PairRecord, is_daily_limit
from dubai_planner.evaluation.scenarios import (
    MIN_SCENARIOS,
    ExpectedOutcome,
    Scenario,
    load_scenarios,
)
from dubai_planner.llm.mock import MockLLM
from dubai_planner.models import (
    ActivityKind,
    DietaryFlag,
    PlannerMode,
    PlanResult,
    PlanStatus,
)
from dubai_planner.storage import Repository

CATALOG = load_catalog()
TOL = Decimal("0.05")


def result(it: Any = None, status: PlanStatus = PlanStatus.APPROVED, **req: Any) -> PlanResult:
    return PlanResult(session_id="s1", run_id="r1", mode=PlannerMode.LANGGRAPH, status=status,
                      request=trip_request(**req), itinerary=it or itinerary(days=3))


def scenario(**expect: Any) -> Scenario:
    return Scenario.model_validate({"id": "t", "title": "t", "category": "test",
                                    "turns": ["hello"], "expect": expect})


# ------------------------------------------------------------------ scenarios


def test_benchmark_has_at_least_twelve_valid_scenarios_covering_the_plan() -> None:
    scenarios = load_scenarios()
    assert len(scenarios) >= MIN_SCENARIOS
    assert len({s.id for s in scenarios}) == len(scenarios)
    categories = {s.category for s in scenarios}
    for needed in ("budget", "family", "luxury", "summer-heat", "ramadan", "constraint-heavy",
                   "day-trip", "tight-budget", "vague", "refinement"):
        assert needed in categories, needed
    veg = next(s for s in scenarios if s.id == "vegetarian_wheelchair_july")
    assert veg.expect.dietary and veg.expect.accessibility and veg.expect.heat_safe
    assert any(s.multi_turn and len(s.turns) >= 3 for s in scenarios)  # multi-refinement
    assert next(s for s in scenarios if s.category == "tight-budget").expect.outcome is (
        ExpectedOutcome.INFEASIBLE_EXPLAINED)


def test_scenario_filter_and_validation() -> None:
    assert [s.id for s in load_scenarios(ids=["ramadan"])] == ["ramadan"]
    with pytest.raises(ValueError, match="unknown scenario"):
        load_scenarios(ids=["nope"])
    with pytest.raises(ValueError):
        Scenario.model_validate({"id": "x", "title": "x", "category": "c", "turns": ["  "]})


# ------------------------------------------------------------------ constraints


def test_constraints_pass_on_a_compliant_plan() -> None:
    checks = check_constraints(scenario(days=3, adults=2, children=1, dietary=["vegetarian"],
                                        heat_safe=True, family_friendly=True,
                                        latest_end="19:30"),
                               result(), CATALOG)
    assert {c.name for c in checks} == {"days", "travellers", "dietary", "heat_safe",
                                        "family_friendly", "latest_end"}
    assert all(c.satisfied for c in checks), [c for c in checks if not c.satisfied]


def test_constraints_fail_with_the_reason() -> None:
    it = itinerary(days=2)
    meat = next(r for r in CATALOG.restaurants if not r.caters_for([DietaryFlag.VEGETARIAN]))
    day = it.days[0]
    bad = [slot("09:00", "10:30"), slot("12:00", "13:30", ref_id="att_dubai_fountain",
                                       indoor=False),
           slot("19:00", "22:30", ref_id=meat.id, kind=ActivityKind.DINING)]
    it = it.model_copy(update={"days": [day.model_copy(update={"slots": bad}), *it.days[1:]]})
    checks = {c.name: c for c in check_constraints(
        scenario(days=3, adults=3, dietary=["vegetarian"], heat_safe=True,
                 latest_end="21:30", day_trip_area="abu_dhabi", pace="relaxed"),
        result(it), CATALOG)}
    assert not checks["days"].satisfied and "2 planned" in checks["days"].detail
    assert not checks["travellers"].satisfied
    assert not checks["dietary"].satisfied and meat.id in checks["dietary"].detail
    assert not checks["heat_safe"].satisfied and "12:00" in checks["heat_safe"].detail
    assert not checks["latest_end"].satisfied and "22:30" in checks["latest_end"].detail
    assert not checks["day_trip"].satisfied
    assert checks["pace"].satisfied  # 2 sightseeing slots ≤ 3


def test_indoor_day_needs_only_indoor_attractions() -> None:
    checks = check_constraints(scenario(indoor_days=[1]), result(), CATALOG)
    assert [(c.name, c.satisfied) for c in checks] == [("indoor_day_1", False)]
    assert checks[0].detail == "1/2 attractions indoor"


# ------------------------------------------------------------------ scoring


def turn(res: PlanResult | None, error: str | None = None) -> TurnOutcome:
    return TurnOutcome(message="m", routing="new_request", explanation="", result=res,
                       trace=None, error=error, wall_s=1.5)


def test_score_within_budget_is_compliant_and_valid() -> None:
    m = score(scenario(days=3, budget_aed="6000"), [turn(result())], mode="langgraph", seed=0,
              catalog=CATALOG, tolerance=TOL)
    assert m.schema_valid and m.within_tolerance and m.budget_compliant and m.outcome_ok
    assert m.constraint_rate == 1.0 and m.grounding_rate == 1.0
    assert m.overshoot_pct is not None and m.overshoot_pct < 0
    assert m.latency_s == 1.5 and m.mean_daily_km is not None


def test_score_counts_overspend_against_the_users_budget() -> None:
    """The user said AED 1000: a plan costing more is non-compliant, even if parsed wrongly."""
    m = score(scenario(budget_aed="1000"), [turn(result())], mode="langgraph", seed=0,
              catalog=CATALOG, tolerance=TOL)
    assert m.within_tolerance is False and m.budget_compliant is False and not m.outcome_ok


def test_score_accepts_an_explained_infeasible_budget() -> None:
    it = itinerary(days=3)
    it = it.model_copy(update={"notes": ["Estimated cost AED 1,224.00 exceeds the budget "
                                         "AED 400.00 (limit …)"]})
    m = score(scenario(outcome="infeasible_explained", budget_aed="400"),
              [turn(result(it, status=PlanStatus.NEEDS_REVISION))], mode="langgraph", seed=0,
              catalog=CATALOG, tolerance=TOL)
    assert m.infeasibility_explained and m.budget_compliant and m.outcome_ok


def test_score_failed_and_clarification_cells() -> None:
    failed = score(scenario(), [turn(None, error="boom")], mode="langgraph", seed=0,
                   catalog=CATALOG, tolerance=TOL)
    assert failed.failed and failed.schema_valid is False and failed.error == "boom"
    asked = PlanResult(session_id="s", run_id="r", mode=PlannerMode.LANGGRAPH,
                       status=PlanStatus.NEEDS_CLARIFICATION)
    vague = score(scenario(asks_clarification=True, budget_aed="6000"),
                  [turn(asked), turn(result())], mode="langgraph", seed=0, catalog=CATALOG,
                  tolerance=TOL)
    assert vague.outcome_ok and len(vague.turns) == 2
    skipped = score(scenario(asks_clarification=True, budget_aed="6000"), [turn(result())],
                    mode="langgraph", seed=0, catalog=CATALOG, tolerance=TOL)
    assert not skipped.outcome_ok  # never asked the expected question


def test_stats_are_mean_and_sample_std_skipping_missing() -> None:
    s = stat([True, False, None, True])
    assert s.n == 3 and s.mean == pytest.approx(2 / 3) and s.std == pytest.approx(0.57735, 1e-4)
    assert stat([None]).mean is None and stat([None]).show() == "—"
    cells = [CellMetrics(scenario_id="a", category="c", mode="langgraph", latency_s=1.0),
             CellMetrics(scenario_id="b", category="c", mode="langgraph", latency_s=3.0)]
    assert aggregate(cells)["langgraph"]["latency_s"].show(1) == "2.0 ± 1.4 (n=2)"


def test_recorded_errors_hide_account_ids() -> None:
    from dubai_planner.evaluation.metrics import scrub

    text = "429 Rate limit for model x in organization `org_01abcdef123` on tokens per day"
    assert scrub(text) == "429 Rate limit for model x in organization `org_***` on tokens per day"
    assert scrub(None) is None


def test_daily_quota_errors_are_recognised() -> None:
    assert is_daily_limit("RateLimitError: Error code: 429 - Limit 200000, Used 199000 on "
                          "tokens per day (TPD)")
    assert not is_daily_limit("Error code: 429 - tokens per minute (TPM): Limit 8000")
    assert not is_daily_limit(None)


# ------------------------------------------------------------------ judge


def test_mock_judge_is_a_constant_placeholder() -> None:
    scores = judge.rubric(MockLLM(), trip_request(), itinerary())
    assert scores.mean == 3.0
    pair = judge.pairwise(MockLLM(), trip_request(), itinerary(), itinerary())
    assert pair.winner == "tie" and pair.consistent


def test_pairwise_counts_only_order_independent_wins() -> None:
    a, b = itinerary(), itinerary(days=3)
    llm = MockLLM().script("judge", judge.PairwiseVerdict(winner="A"),
                           judge.PairwiseVerdict(winner="B"))
    res = judge.pairwise(llm, trip_request(), a, b)
    assert (res.first, res.swapped, res.winner, res.consistent) == ("A", "a", "a", True)
    biased = MockLLM().script("judge", judge.PairwiseVerdict(winner="A"),
                              judge.PairwiseVerdict(winner="A"))
    res = judge.pairwise(biased, trip_request(), a, b)
    assert res.winner == "tie" and not res.consistent  # position bias, not a win


# ------------------------------------------------------------------ report + loader


def outcome(provider: str = "mock") -> EvalOutcome:
    cells = [CellMetrics(scenario_id=s, category="c", mode=m, latency_s=lat, llm_calls=5,
                         tokens=1000, schema_valid=True, budget_compliant=True,
                         constraint_rate=1.0, outcome_ok=True, interest_match=0.5)
             for s in ("a", "b") for m, lat in (("langgraph", 1.0), ("crewai_sequential", 2.0))]
    faults = [FaultResult(defect="over_budget", rule_id="budget.total", severity="error",
                          mode=m, injected=True, present_in_attempt=True, detected=True,
                          detected_by=["review:review"], fixed=True, final_status="approved",
                          iterations=1) for m in ("langgraph", "crewai_sequential")]
    cfg = EvalConfig(provider=provider, model="mock", modes=[PlannerMode.LANGGRAPH,
                                                             PlannerMode.CREWAI_SEQUENTIAL],
                     scenarios=["a", "b"], label="unit")
    return EvalOutcome(eval_run_id="e1", config=cfg, cells=cells, faults=faults,
                       pairs=[PairRecord(scenario_id="a", seed=0, a="langgraph",
                                         b="crewai_sequential", winner="tie", consistent=True,
                                         first="tie", swapped="tie")])


def test_report_files_and_round_trip(tmp_path: Path) -> None:
    out_dir = tmp_path / "results" / "unit"
    report = write_report(outcome(), out_dir)
    text = report.read_text(encoding="utf-8")
    assert MOCK_BANNER in text and "## Summary per mode" in text
    assert "detection rate **1.00**" in text
    for name in ("results.csv", "cells.jsonl", "config.json", "faults.csv", "pairwise.jsonl",
                 "performance.png", "quality.png"):
        assert (out_dir / name).is_file(), name
    assert (out_dir / "results.csv").read_text(encoding="utf-8").count("\n") == 5
    [ds] = load_published(tmp_path / "results")
    assert ds.is_mock and len(ds.cells) == 4 and ds.report_path == report
    assert ds.faults == outcome().faults and len(ds.pairs) == 1
    assert detection_rate(ds.faults) == 1.0 and fix_rate(ds.faults, "langgraph") == 1.0


def test_real_provider_report_has_no_mock_banner(tmp_path: Path) -> None:
    text = write_report(outcome("groq"), tmp_path).read_text(encoding="utf-8")
    assert MOCK_BANNER not in text and "Real provider" in text


def test_eval_runs_are_listed_from_the_database(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "e.db")
    first = repo.create_eval_run({"label": "x", "provider": "mock"})
    assert repo.get_eval_run(first.id) == first
    assert repo.get_eval_run("missing") is None
    assert [r.id for r in repo.list_eval_runs()] == [first.id]
    [ds] = load_database(tmp_path / "e.db")
    assert ds.cells == [] and ds.is_mock
    assert load_database(tmp_path / "absent.db") == []


# ------------------------------------------------------------------ deployment helpers


def test_project_root_can_be_set_explicitly(monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path) -> None:
    from dubai_planner import config

    monkeypatch.setenv("DUBAI_PLANNER_ROOT", str(tmp_path))
    assert config._project_root() == tmp_path.resolve()
    monkeypatch.delenv("DUBAI_PLANNER_ROOT")
    assert (config._project_root() / "data" / "seed").is_dir()


def test_streamlit_secrets_become_environment_variables(monkeypatch: pytest.MonkeyPatch
                                                        ) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "app"))
    import ui_shared

    monkeypatch.setenv("LLM_PROVIDER", "mock")  # an explicit env var wins
    monkeypatch.delenv("HTTP_OFFLINE", raising=False)
    fake = SimpleNamespace(secrets={"LLM_PROVIDER": "groq", "http_offline": True,
                                    "nested": {"x": 1}})
    monkeypatch.setattr(ui_shared, "st", fake)
    ui_shared._secrets_to_env()
    import os

    assert os.environ["LLM_PROVIDER"] == "mock"
    assert os.environ["HTTP_OFFLINE"] == "True"
    assert "NESTED" not in os.environ
    monkeypatch.delenv("HTTP_OFFLINE")

    class NoSecrets:
        @property
        def secrets(self) -> Any:
            raise FileNotFoundError("no secrets.toml")

    monkeypatch.setattr(ui_shared, "st", NoSecrets())
    ui_shared._secrets_to_env()  # no secrets file: silently nothing
    sys.modules.pop("ui_shared", None)


def test_heat_slot_overlap_uses_the_shared_midday_window() -> None:
    from dubai_planner.evaluation.metrics import _overlaps_midday

    assert _overlaps_midday(time(10, 30), time(11, 30))
    assert not _overlaps_midday(time(16, 0), time(17, 0))
