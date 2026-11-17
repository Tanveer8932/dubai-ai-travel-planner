"""Phase 7 end to end (mock provider, offline): eval runner, fault injection, transcript.

These are orchestration checks. They prove the evaluation machinery and the review
loop behave as specified; they are not evidence of real-model quality.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from dubai_planner.config import Settings
from dubai_planner.evaluation.faults import DEFECTS, detection_rate, fix_rate, run_fault
from dubai_planner.evaluation.runner import EvalRunner, make_config
from dubai_planner.evaluation.scenarios import load_scenarios
from dubai_planner.evaluation.transcript import record_session, snapshot, verify
from dubai_planner.llm.mock import MockLLM
from dubai_planner.models import PlannerMode
from dubai_planner.storage import Repository
from dubai_planner.tools.toolbox import Toolbox

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("HTTP_OFFLINE", "true")


# ------------------------------------------------------------------ AC7.1


def test_ac71_run_eval_offline_writes_csv_and_markdown(offline: None, tmp_path: Path) -> None:
    run_eval = load_script("run_eval")
    out = tmp_path / "out"
    code = run_eval.main(["--provider", "mock", "--scenarios", "budget_backpacker",
                          "multi_refinement", "--judge", "--no-faults", "--no-warmup",
                          "--db", str(tmp_path / "eval.db"), "--out", str(out)])
    assert code == 0
    with (out / "results.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {(r["scenario_id"], r["mode"]) for r in rows} == {
        (s, m) for s in ("budget_backpacker", "multi_refinement")
        for m in ("langgraph", "crewai_sequential")}
    assert all(r["schema_valid"] == "True" and r["final_status"] == "approved" for r in rows)
    refine = [r for r in rows if r["scenario_id"] == "multi_refinement"]
    assert all(r["routing"] == "new_request > request_delta > local_edit" for r in refine)
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Mock provider" in report and "## Summary per mode" in report
    assert "± " in report and "## Pairwise judge" in report
    assert (out / "performance.png").stat().st_size > 1000


def test_resume_and_token_budget(offline: None, tmp_path: Path) -> None:
    settings = Settings()
    repo = Repository(tmp_path / "eval.db")
    scenarios = load_scenarios(ids=["budget_backpacker", "ramadan"])
    modes = [PlannerMode.LANGGRAPH, PlannerMode.CREWAI_SEQUENTIAL]
    runner = EvalRunner(repo, settings=settings, llm=MockLLM(),
                        checkpoint_path=tmp_path / "cp.db")
    first = runner.evaluate(scenarios, make_config(settings, modes=modes, scenarios=scenarios,
                                                   max_tokens=1, warmup=False))
    assert len(first.cells) == 1 and first.stopped_reason
    assert first.stopped_reason.startswith("token budget reached")
    assert len(first.skipped) == 3
    cfg = make_config(settings, modes=modes, scenarios=scenarios, warmup=False)
    second = EvalRunner(repo, settings=settings, llm=MockLLM(),
                        checkpoint_path=tmp_path / "cp.db").evaluate(
        scenarios, cfg, resume=first.eval_run_id)
    keys = sorted((c.scenario_id, c.mode) for c in second.cells)
    assert len(keys) == 4 and len(set(keys)) == 4  # finished cell not re-run
    assert len(repo.list_eval_results(first.eval_run_id)) == 4
    with pytest.raises(ValueError, match="no evaluation run"):
        runner.evaluate(scenarios, cfg, resume="missing")


def test_every_scenario_produces_a_valid_result_in_langgraph(offline: None,
                                                              tmp_path: Path) -> None:
    """S1 on the whole benchmark (mock): every cell yields a schema-valid itinerary."""
    settings = Settings()
    scenarios = load_scenarios()
    runner = EvalRunner(Repository(tmp_path / "eval.db"), settings=settings, llm=MockLLM(),
                        checkpoint_path=tmp_path / "cp.db")
    out = runner.evaluate(scenarios, make_config(settings, modes=[PlannerMode.LANGGRAPH],
                                                 scenarios=scenarios, warmup=False))
    assert len(out.cells) == len(scenarios)
    assert all(c.schema_valid for c in out.cells if c.itinerary_expected)
    assert not any(c.failed for c in out.cells)
    by_id = {c.scenario_id: c for c in out.cells}
    assert by_id["tight_budget_infeasible"].infeasibility_explained
    assert by_id["vague_request"].outcome_ok
    assert by_id["solo_usd_budget"].budget_aed == pytest.approx(5508.75)
    assert by_id["vegetarian_wheelchair_july"].constraint_rate == 1.0


# ------------------------------------------------------------------ AC7.4


@pytest.mark.parametrize("mode", [PlannerMode.LANGGRAPH, PlannerMode.CREWAI_SEQUENTIAL],
                         ids=lambda m: m.value)
def test_ac74_review_loop_detects_seeded_defects(mode: PlannerMode, offline: None,
                                                 tmp_path: Path) -> None:
    repo, tools = Repository(tmp_path / "f.db"), Toolbox.default()
    results = [run_fault(d, mode, repo=repo, tools=tools, checkpoint_path=tmp_path / "cp.db",
                         llm=MockLLM()) for d in DEFECTS]
    assert all(r.injected and r.present_in_attempt for r in results), [
        (r.defect, r.error) for r in results if not r.present_in_attempt]
    rate = detection_rate(results)
    assert rate is not None and rate >= 0.8
    assert fix_rate(results) == 1.0  # every error-severity defect repaired and approved
    for r in results:
        assert r.detected, r
        if r.severity == "error":
            assert r.iterations >= 1 and r.final_status == "approved"


# ------------------------------------------------------------------ transcript


def test_transcript_conversation_survives_a_restart(offline: None, tmp_path: Path) -> None:
    db, cp = tmp_path / "t.db", tmp_path / "t_cp.db"
    log = record_session(db, cp, PlannerMode.LANGGRAPH, llm=MockLLM())
    kinds = [t.routing for t in log.turns]
    assert kinds == ["new_request", "request_delta", "local_edit", "general_change"]
    assert [t.saved_version for t in log.turns] == [1, 2, 3, 4]
    budget_cut, day_edit, veg = log.turns[1], log.turns[2], log.turns[3]
    assert budget_cut.budget_aed == "4800.00"
    assert day_edit.locked_days == [1, 3] and day_edit.diff is not None
    assert day_edit.diff["changed_days"] == [2] and day_edit.diff["unchanged_days"] == [1, 3]
    assert veg.dietary == ["vegetarian"] and veg.vegetarian_restaurants_only
    assert log.before_close is not None
    reopened = snapshot(Repository(db), log.session_id, checkpoint_path=cp)
    checks = verify(log.before_close, reopened)
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]


def test_transcript_script_reopens_in_a_new_process(offline: None, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script("session_transcript")
    code = script.main(["--mode", "crewai_sequential", "--out", str(tmp_path / "out"),
                        "--db", str(tmp_path / "s.db")])
    assert code == 0
    md = (tmp_path / "out" / "session_crewai_sequential.md").read_text(encoding="utf-8")
    assert "| chat history identical | PASS |" in md and "FAIL" not in md
    import json

    data: dict[str, Any] = json.loads(
        (tmp_path / "out" / "session_crewai_sequential.json").read_text(encoding="utf-8"))
    assert data["reopened_in_pid"] != data["parent_pid"]
