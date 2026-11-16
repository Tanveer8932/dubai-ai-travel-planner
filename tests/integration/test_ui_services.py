"""Phase 5 services: sessions, version diff, exports, price provenance, runs, refinement.

Planner runs go through ``RunManager`` → ``PlannerHub`` → the real planners (mock LLM),
as the Streamlit pages do.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from planner_helpers import TODAY, WINTER

from dubai_planner.config import Settings
from dubai_planner.llm.mock import MockLLM
from dubai_planner.models import ActivityKind, PlannerMode, PlanStatus, RunStatus
from dubai_planner.services import presentation as view
from dubai_planner.services.refinement import ChangeKind
from dubai_planner.services.runs import (
    PlannerHub,
    RunHandle,
    RunInProgress,
    RunManager,
    friendly_error,
    redact,
)
from dubai_planner.services.session import DEFAULT_TITLE, SessionService
from dubai_planner.storage import Repository
from dubai_planner.tools.toolbox import Toolbox

MODES = [PlannerMode.LANGGRAPH, PlannerMode.CREWAI_SEQUENTIAL]


@pytest.fixture
def stack(tmp_path: Path, toolbox: Toolbox) -> Iterator[tuple[SessionService, RunManager]]:
    repo = Repository(tmp_path / "app.db")
    sessions = SessionService(repo)
    hub = PlannerHub(repo, checkpoint_path=tmp_path / "cp.db", tools=toolbox, llm=MockLLM())
    yield sessions, RunManager(hub, sessions, today=lambda: TODAY)
    hub.close()


def plan(runs: RunManager, session_id: str, mode: PlannerMode, text: str) -> RunHandle:
    handle = runs.start(session_id, mode, text)
    assert handle.wait(120), "run did not finish"
    assert handle.error is None, handle.error
    return handle


# ------------------------------------------------------------------ AC5.3 (services)


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
def test_ac53_budget_cut_and_day_edit(mode: PlannerMode, stack: Any) -> None:
    sessions, runs = stack
    s = sessions.create(mode)
    plan(runs, s.id, mode, WINTER)
    cut = plan(runs, s.id, mode, "reduce the budget by 20%")
    assert cut.refinement.kind is ChangeKind.REQUEST_DELTA
    assert cut.result is not None and cut.result.status is PlanStatus.APPROVED
    v1, v2 = sessions.versions(s.id)
    assert v2.request is not None and v2.request.total_budget_aed() == Decimal("4800")
    assert v2.itinerary.cost.total <= Decimal("0.8") * Decimal("6000") * Decimal("1.05")
    assert v2.parent_id == v1.id
    edit = plan(runs, s.id, mode, "Replace day 2 with indoor activities")
    assert edit.refinement.kind is ChangeKind.LOCAL_EDIT
    v3 = sessions.latest(s.id)
    assert v3 is not None and v3.version == 3 and v3.request == v2.request
    diff = sessions.diff(v2, v3)
    assert diff.changed_days == [1] and diff.unchanged_days == [0, 2]
    assert v3.itinerary.days[0] == v2.itinerary.days[0]
    assert v3.itinerary.days[2] == v2.itinerary.days[2]
    attractions = [sl for sl in v3.itinerary.days[1].slots
                   if sl.activity.kind is ActivityKind.ATTRACTION]
    assert attractions and all(sl.indoor for sl in attractions)
    kept = {sl.activity.ref_id for d in (0, 2) for sl in v3.itinerary.days[d].slots
            if sl.activity.kind is ActivityKind.ATTRACTION}
    assert not kept & {sl.activity.ref_id for sl in attractions}  # nothing repeated
    assert [v.version for v in sessions.versions(s.id)] == [1, 2, 3]


def test_local_edit_reuses_the_request_without_an_llm_call(stack: Any,
                                                            tmp_path: Path) -> None:
    sessions, runs = stack
    mock = runs.hub.planner_kwargs["llm"]
    s = sessions.create(PlannerMode.LANGGRAPH)
    plan(runs, s.id, PlannerMode.LANGGRAPH, WINTER)
    before = sum(1 for c in mock.calls if c.agent == "requirements")
    plan(runs, s.id, PlannerMode.LANGGRAPH, "Replace day 2 with indoor activities")
    assert sum(1 for c in mock.calls if c.agent == "requirements") == before


def test_refinement_works_across_modes(stack: Any) -> None:
    """A session started in LangGraph can be refined in CrewAI (state is in SQLite)."""
    sessions, runs = stack
    s = sessions.create(PlannerMode.LANGGRAPH)
    plan(runs, s.id, PlannerMode.LANGGRAPH, WINTER)
    handle = plan(runs, s.id, PlannerMode.CREWAI_SEQUENTIAL, "reduce the budget by 20%")
    assert handle.result is not None and handle.result.status is PlanStatus.APPROVED
    latest = sessions.latest(s.id)
    assert latest is not None and latest.mode is PlannerMode.CREWAI_SEQUENTIAL
    assert latest.request is not None and latest.request.total_budget_aed() == Decimal("4800")


def test_changed_dates_drop_day_locks(stack: Any) -> None:
    from factories import itinerary, trip_request

    from dubai_planner.agents.revision import applicable_locks

    previous = itinerary(days=3)
    same = trip_request(end_date=date(2027, 2, 12))
    longer = trip_request(end_date=date(2027, 2, 13))
    moved = trip_request(start_date=date(2027, 3, 10), end_date=date(2027, 3, 12))
    assert applicable_locks([0, 2], previous, same) == [0, 2]
    assert applicable_locks([0, 2], previous, longer) == []
    assert applicable_locks([0, 2], previous, moved) == []
    assert applicable_locks([0, 5], previous, same) == [0]


# ------------------------------------------------------------------ runs


def test_one_active_run_per_session(stack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, runs = stack
    gate = threading.Event()
    original = RunManager._run
    monkeypatch.setattr(RunManager, "_run",
                        lambda self, h: (gate.wait(30), original(self, h)))
    s = sessions.create(PlannerMode.LANGGRAPH)
    first = runs.start(s.id, PlannerMode.LANGGRAPH, WINTER)
    with pytest.raises(RunInProgress):
        runs.start(s.id, PlannerMode.CREWAI_SEQUENTIAL, WINTER)
    other = sessions.create(PlannerMode.LANGGRAPH)  # other sessions are independent
    second = runs.start(other.id, PlannerMode.LANGGRAPH, WINTER)
    gate.set()
    assert first.wait(120) and second.wait(120)
    assert runs.active(s.id) is None and runs.last(s.id) is first
    with pytest.raises(ValueError):
        runs.start(s.id, PlannerMode.LANGGRAPH, "   ")


def test_planner_crash_becomes_a_friendly_error(stack: Any,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, runs = stack

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Error code: 401 - invalid api key")

    monkeypatch.setattr(runs.hub, "planner", lambda _mode: type("P", (), {
        "stream": boom, "pending_question": lambda self, _s: None})())
    s = sessions.create(PlannerMode.LANGGRAPH)
    handle = runs.start(s.id, PlannerMode.LANGGRAPH, WINTER)
    assert handle.wait(30)
    assert handle.user_error is not None and "rejected the API key" in handle.user_error


@pytest.mark.parametrize(("error", "expected"), [
    ("LLMConfigError: LLM_PROVIDER=groq requires GROQ_API_KEY to be set", "GROQ_API_KEY"),
    ("LLMCallError: groq: AuthenticationError: Error code: 401", "rejected the API key"),
    ("LLMCallError: groq: RateLimitError: Error code: 429", "rate limit"),
    ("LLMCallError: groq: APIConnectionError: Connection error.", "could not be reached"),
    ("StructuredOutputError: composer: output did not validate", "could not be validated"),
    ("RuntimeError: something odd", "Planning failed: RuntimeError"),
])
def test_friendly_errors(error: str, expected: str) -> None:
    assert expected in friendly_error(error)


def test_redact_removes_configured_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    canary = "gsk_live_like_secret_value_123"
    monkeypatch.setenv("GROQ_API_KEY", canary)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert redact(f"failed with key {canary}", s) == "failed with key ***"
    assert canary not in friendly_error(f"LLMCallError: 401 key={canary}", s)


def test_failed_run_is_recorded(stack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, runs = stack
    from dubai_planner.agents.suite import AgentSuite
    from dubai_planner.llm.errors import StructuredOutputError

    def broken(*_a: Any, **_k: Any) -> Any:
        raise StructuredOutputError("dining", "Selection", ["bad"], "{")

    runs.hub.planner_kwargs["agents"] = AgentSuite(dining=broken)
    s = sessions.create(PlannerMode.CREWAI_SEQUENTIAL)
    handle = plan(runs, s.id, PlannerMode.CREWAI_SEQUENTIAL, WINTER)
    assert handle.user_error is not None and "could not be validated" in handle.user_error
    assert sessions.runs(s.id)[-1].status is RunStatus.FAILED


# ------------------------------------------------------------------ sessions


def test_session_service_lifecycle(stack: Any) -> None:
    sessions, runs = stack
    s = sessions.create(PlannerMode.CREWAI_HIERARCHICAL)
    assert s.title == DEFAULT_TITLE and s.mode is PlannerMode.CREWAI_HIERARCHICAL
    plan(runs, s.id, PlannerMode.LANGGRAPH, WINTER)
    assert sessions.get(s.id).title.startswith("2 adults, 10-12 Feb")  # type: ignore[union-attr]
    sessions.rename(s.id, "  Winter  ")
    assert sessions.get(s.id).title == "Winter"  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        sessions.rename(s.id, "   ")
    assert [m.role.value for m in sessions.history(s.id)] == ["user", "assistant"]
    assert sessions.version(s.id, 1) is not None and sessions.version(s.id, 9) is None
    runs.hub.forget_thread(s.id)
    graph = runs.hub.planner(PlannerMode.LANGGRAPH).graph
    assert not graph.get_state({"configurable": {"thread_id": s.id}}).values
    assert sessions.delete(s.id) and sessions.get(s.id) is None
    assert sessions.versions(s.id) == [] and sessions.history(s.id) == []


# ------------------------------------------------------------------ AC5.6 / exports


def test_every_slot_price_has_an_estimate_label_and_source(stack: Any,
                                                           toolbox: Toolbox) -> None:
    sessions, runs = stack
    s = sessions.create(PlannerMode.LANGGRAPH)
    plan(runs, s.id, PlannerMode.LANGGRAPH, WINTER)
    record = sessions.latest(s.id)
    assert record is not None
    for day in record.itinerary.days:
        for slot in day.slots:
            note = view.venue_price(slot, toolbox.catalog)
            if slot.cost_aed > 0:
                assert "(estimate)" in note.label and "as of" in note.source
                assert "estimate" in note.basis
            if slot.transport_from_prev is not None:
                fare = view.transport_price(slot.transport_from_prev, toolbox.catalog)
                assert "as of" in fare.source
    assert record.itinerary.hotel is not None
    assert "as of" in view.hotel_price(record.itinerary.hotel).source
    rows = view.cost_rows(record.itinerary)
    assert rows[-1]["Category"] == "Budget" and all("estimate" in k for r in rows
                                                     for k in r if k != "Category")


def test_exports_are_complete_labelled_and_secret_free(stack: Any, toolbox: Toolbox,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, runs = stack
    s = sessions.create(PlannerMode.LANGGRAPH)
    plan(runs, s.id, PlannerMode.LANGGRAPH, WINTER)
    record = sessions.latest(s.id)
    assert record is not None
    data = json.loads(view.export_json(record))
    assert data["prices_are_estimates"] and data["version"] == 1
    assert data["itinerary"]["days"] and data["request"]["budget_aed"]
    md = view.export_markdown(record, toolbox.catalog)
    assert md.startswith("# ") and "## Day 1" in md and "## Budget (AED, estimates)" in md
    assert "(estimate)" in md and "as of" in md and "OpenStreetMap" in md
    text = view.export_json(record) + md
    assert "api_key" not in text.lower() and "gsk_" not in text
