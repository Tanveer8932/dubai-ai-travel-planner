"""AC4.1: the ``PlannerService`` contract, checked identically for every orchestration mode.

The UI and the evaluator only see :class:`Planner`; these tests are the same for
LangGraph, CrewAI sequential and CrewAI hierarchical, plus a parity check that the two
frameworks turn the same request into the same itinerary (one shared agent core).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from planner_helpers import TODAY, WINTER

from dubai_planner.llm.mock import MockLLM
from dubai_planner.models import (
    Itinerary,
    PlannerMode,
    PlannerUpdate,
    PlanResult,
    PlanStatus,
    RunStatus,
)
from dubai_planner.services.planner import Planner, open_planner
from dubai_planner.storage import Repository
from dubai_planner.tools.toolbox import Toolbox

MODES = [PlannerMode.LANGGRAPH, PlannerMode.CREWAI_SEQUENTIAL,
         PlannerMode.CREWAI_HIERARCHICAL]
#: Fields that identify a run rather than describe the plan.
RUN_FIELDS = {"id", "session_id", "mode", "created_at", "version", "parent_id"}


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "app.db")


@pytest.fixture
def planners(tmp_path: Path, toolbox: Toolbox, repo: Repository) -> Iterator[Any]:
    stack = ExitStack()

    def make(mode: PlannerMode) -> Planner:
        return stack.enter_context(open_planner(
            mode, checkpoint_path=tmp_path / f"cp-{mode.value}.db", llm=MockLLM(),
            tools=toolbox, repo=repo))

    yield make
    stack.close()


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
def test_ac41_planner_contract(mode: PlannerMode, planners: Any, repo: Repository) -> None:
    planner = planners(mode)
    assert isinstance(planner, Planner)
    session = f"contract-{mode.value}"
    updates = list(planner.stream(WINTER, session, today=TODAY))
    assert all(isinstance(u, PlannerUpdate) for u in updates)
    assert [u.is_final for u in updates] == [False] * (len(updates) - 1) + [True]
    result = updates[-1].result
    assert isinstance(result, PlanResult) and result.mode is mode
    assert result.status is PlanStatus.APPROVED and result.error is None
    assert all(u.run_id == result.run_id and u.session_id == session for u in updates)
    itinerary, request = result.itinerary, result.request
    assert type(itinerary) is Itinerary and itinerary.mode is mode and request is not None
    assert Itinerary.model_validate_json(itinerary.model_dump_json()) == itinerary
    assert itinerary.cost.total <= request.total_budget_aed() * Decimal("1.05")
    assert result.review is not None and result.review.approved and result.attempts
    assert result.saved_version == 1
    record = repo.latest_itinerary(session)
    assert record is not None and record.approved and record.mode is mode
    run = repo.get_run(result.run_id)
    assert run is not None and run.status is RunStatus.SUCCEEDED and run.llm_calls > 0


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
def test_ac41_clarification_contract(mode: PlannerMode, planners: Any) -> None:
    planner = planners(mode)
    session = f"clarify-{mode.value}"
    asked = planner.run("plan a Dubai trip", session, today=TODAY)
    assert asked.status is PlanStatus.NEEDS_CLARIFICATION and asked.clarification_question
    assert planner.pending_question(session) == asked.clarification_question
    done = planner.run("5 days in Feb, AED 8000", session, today=TODAY)
    assert done.status is PlanStatus.APPROVED and done.run_id == asked.run_id
    assert done.request is not None and done.request.num_days == 5


def test_ac41_both_frameworks_plan_the_same_itinerary(planners: Any) -> None:
    """Same request, same mock → same plan: orchestration differs, business logic does not."""
    results = {mode: planners(mode).run(WINTER, f"parity-{mode.value}", today=TODAY)
               for mode in MODES}
    plans = {mode: r.itinerary.model_dump(mode="json", exclude=RUN_FIELDS)  # type: ignore[union-attr]
             for mode, r in results.items()}
    reference = plans[PlannerMode.LANGGRAPH]
    for mode in MODES[1:]:
        assert plans[mode] == reference, mode
    assert len({r.request.model_dump_json() for r in results.values()  # type: ignore[union-attr]
                }) == 1
