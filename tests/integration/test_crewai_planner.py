"""Phase 4: CrewAI orchestration end to end (offline, mock LLM unless marked ``llm``).

The mock drives real CrewAI machinery: ``Crew.kickoff``, ``Task`` context and
``async_execution``, guardrail retries, the hierarchical manager's delegation tool and
CrewAI's event bus. Mock results prove orchestration behaviour, never real-model
quality; AC4 has no real-model criterion (see ``test_crewai_real_model_smoke``).
"""

from __future__ import annotations

import json
import socket
import threading
from collections import Counter
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from planner_helpers import (
    SUMMER,
    TODAY,
    WINTER,
    ComposerSpy,
    HeatDefectComposer,
    slow,
)

from dubai_planner.agents import attractions, dining
from dubai_planner.agents.suite import AgentSuite
from dubai_planner.llm.errors import LLMCallError, StructuredOutputError
from dubai_planner.llm.mock import MockLLM
from dubai_planner.models import (
    ChatRole,
    Itinerary,
    PlannerMode,
    PlanStatus,
    ReviewReport,
    RunStatus,
    Severity,
    TraceEvent,
    TraceEventType,
)
from dubai_planner.orchestration.common import RetrySettings
from dubai_planner.orchestration.crewai import STAGES, CrewAIPlanner, run_crewai
from dubai_planner.orchestration.crewai.board import RESEARCH_STAGES
from dubai_planner.storage import Repository
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import capture

FAST_RETRY = RetrySettings(max_attempts=2, initial_interval=0.01, backoff_factor=1.0,
                           max_interval=0.01, jitter=False)
Factory = Callable[..., CrewAIPlanner]
ROLE_TOOLS = {s.role_key: s.tool for s in STAGES.values()}


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "app.db")


@pytest.fixture
def make_planner(repo: Repository, toolbox: Toolbox, mock_llm: MockLLM) -> Factory:
    def make(**kwargs: Any) -> CrewAIPlanner:
        kwargs.setdefault("llm", mock_llm)
        kwargs.setdefault("tools", toolbox)
        kwargs.setdefault("repo", repo)
        kwargs.setdefault("retry", FAST_RETRY)
        return CrewAIPlanner(**kwargs)

    return make


def of_kind(events: list[TraceEvent], kind: str, event: TraceEventType | None = None
            ) -> list[TraceEvent]:
    return [e for e in events if e.payload.get("kind") == kind
            and (event is None or e.event is event)]



# ------------------------------------------------------------------ AC4.1


def test_ac41_sequential_crew_produces_the_same_itinerary_type(make_planner: Factory,
                                                               repo: Repository) -> None:
    planner = make_planner(process="sequential")
    with capture() as events:
        updates = list(planner.stream(WINTER, "s-e2e", today=TODAY))
    result = updates[-1].result
    assert result is not None and result.status is PlanStatus.APPROVED and result.approved
    itinerary = result.itinerary
    assert type(itinerary) is Itinerary and result.request is not None
    assert Itinerary.model_validate_json(itinerary.model_dump_json()) == itinerary
    assert itinerary.mode is PlannerMode.CREWAI_SEQUENTIAL and result.mode is itinerary.mode
    assert itinerary.cost.total <= result.request.total_budget_aed() * Decimal("1.05")
    assert not [v for v in result.violations if v.severity is Severity.ERROR]
    # streamed: one update per finished CrewAI task, then the final one
    nodes = [u.node for u in updates]
    assert nodes[0] == "parse_request" and nodes[-1] == "__end__"
    assert {STAGES[k].task for k in STAGES} <= set(nodes)
    # persisted: session, conversation, approved v1, run record
    record = repo.latest_itinerary("s-e2e")
    assert record is not None and record.version == 1 and record.approved
    assert record.run_id == result.run_id and result.saved_version == 1
    assert [m.role for m in repo.chat_history("s-e2e")] == [ChatRole.USER, ChatRole.ASSISTANT]
    run = repo.get_run(result.run_id)
    assert run is not None and run.status is RunStatus.SUCCEEDED and run.mode is itinerary.mode
    assert run.llm_calls >= 9 + 7  # crew-agent reasoning calls + agent-core calls
    assert events and all(e.run_id == result.run_id and e.mode == "crewai_sequential"
                          for e in events)


def test_crew_tasks_carry_typed_outputs_and_context(make_planner: Factory, toolbox: Toolbox,
                                                    mock_llm: MockLLM) -> None:
    """Real CrewAI tasks: typed ``TaskOutput.pydantic`` and declared ``context`` chains."""
    from dubai_planner.orchestration.crewai.board import CrewRunBoard
    from dubai_planner.orchestration.crewai.crew import PlanningCrew

    planner = make_planner()
    board = CrewRunBoard(session_id="s-x", run_id="r-x", mode=planner.mode, today=TODAY,
                         user_message=WINTER)
    req = planner.factory.requirements_crew(board)
    req.kickoff(board)
    built = planner.factory.planning_crew(board)
    assert isinstance(built.crew, PlanningCrew)
    tasks = built.tasks
    assert all(tasks[k].async_execution for k in RESEARCH_STAGES)
    assert tasks["budget"].context == [tasks[k] for k in RESEARCH_STAGES]
    assert tasks["compose"].context[0] is tasks["budget"]
    assert tasks["review"].context == [tasks["compose"]]
    out = built.kickoff(board)
    for key, task in tasks.items():
        assert task.output is not None and isinstance(task.output.pydantic, STAGES[key].output)
        assert task.output.agent == built.agents[key].role
        assert len(task.output.raw) < 1500  # short briefs keep downstream prompts small
    assert isinstance(out.pydantic, ReviewReport)
    assert "Budget AED" in tasks["compose"].prompt_context  # budget brief reached the architect


# ------------------------------------------------------------------ AC4.2


def test_ac42_trace_attributes_tasks_and_tool_calls_to_roles(make_planner: Factory) -> None:
    with capture() as events:
        result = make_planner(process="sequential").run(WINTER, "s-roles", today=TODAY)
    assert result.status is PlanStatus.APPROVED
    tasks = of_kind(events, "crew_task", TraceEventType.END)
    assert {e.agent for e in tasks} == {s.task for s in STAGES.values()}
    for e in tasks:  # each task ran on its own role's agent
        stage = next(s for s in STAGES.values() if s.task == e.agent)
        assert e.payload["role_key"] == stage.role_key
        assert e.payload["executed_by"] == next(
            t.payload["role"] for t in of_kind(events, "tool")
            if t.agent == stage.role_key)
    tool_calls = {(e.agent, e.payload["tool"]) for e in of_kind(events, "tool")}
    assert tool_calls == set(ROLE_TOOLS.items())
    reasoning = Counter(e.agent for e in of_kind(events, "crew_agent"))
    assert set(reasoning) == set(ROLE_TOOLS)  # every role reasoned with its own LLM
    assert all(e.tokens_in > 0 for e in of_kind(events, "crew_agent"))
    assert not of_kind(events, "delegation")


def test_ac42_hierarchical_manager_delegates_every_task(make_planner: Factory,
                                                        mock_llm: MockLLM) -> None:
    with capture() as events:
        result = make_planner(process="hierarchical").run(WINTER, "s-hier", today=TODAY)
    assert result.status is PlanStatus.APPROVED
    assert result.itinerary is not None
    assert result.itinerary.mode is PlannerMode.CREWAI_HIERARCHICAL
    delegations = of_kind(events, "delegation")
    assert all(d.agent == "trip_manager" for d in delegations)
    roles = {s.role_key: s for s in STAGES.values()}
    from dubai_planner.orchestration.crewai.crew import role_of

    assert sorted(d.payload["coworker"] for d in delegations) == sorted(
        role_of(k) for k in roles)
    # the specialist the manager delegated to is the one that used the tool
    assert {(e.agent, e.payload["tool"]) for e in of_kind(events, "tool")} == set(
        ROLE_TOOLS.items())
    tasks = of_kind(events, "crew_task", TraceEventType.END)
    assert {e.payload["executed_by"] for e in tasks} == {"Trip Manager"}
    # two manager calls per task (delegate, report): no re-typing of specialist output
    assert Counter(mock_llm.react_calls)["Trip Manager"] == 2 * len(STAGES)


def test_ac42_supporting_tools_are_attributed_to_the_calling_agent(
        make_planner: Factory, mock_llm: MockLLM) -> None:
    role = "Transport & Logistics Specialist"
    mock_llm.script(f"crewai:{role}",
                    'Thought: check a fare first.\nAction: metro_fare\nAction Input: '
                    '{"zones": 2}',
                    "Thought: now the plan.\nAction: plan_transport\nAction Input: {}")
    with capture() as events:
        result = make_planner().run(WINTER, "s-support", today=TODAY)
    assert result.status is PlanStatus.APPROVED
    transport_tools = [e.payload["tool"] for e in of_kind(events, "tool")
                       if e.agent == "transport_specialist"]
    assert transport_tools == ["metro_fare", "plan_transport"]
    assert all(e.payload["ok"] for e in of_kind(events, "tool"))


def test_agent_answering_without_its_tool_is_rejected_and_retried(
        make_planner: Factory, mock_llm: MockLLM) -> None:
    """Hard rule 5: an agent may not invent results; the guardrail sends it back."""
    mock_llm.script("crewai:Dining Specialist",
                    "Thought: I know Dubai well.\nFinal Answer: res_invented_place, "
                    "res_another_fake")
    with capture() as events:
        result = make_planner().run(WINTER, "s-invent", today=TODAY)
    assert result.status is PlanStatus.APPROVED
    rejected = [e for e in of_kind(events, "guardrail")
                if e.agent == "research_dining" and not e.payload["passed"]]
    assert len(rejected) == 1 and rejected[0].payload["reason"] == "no_tool_result"
    assert result.itinerary is not None
    assert not {"res_invented_place", "res_another_fake"} & result.itinerary.venue_ids()
    done = [e for e in of_kind(events, "crew_task", TraceEventType.END)
            if e.agent == "research_dining"]
    assert done[0].payload["retries"] == 1


# ------------------------------------------------------------------ AC4.3


def test_ac43_async_research_tasks_overlap(make_planner: Factory) -> None:
    """Research agents stubbed to sleep 1 s: stage < 1.8 s (vs ≈ 5 s one after another)."""
    probe = make_planner(persist=False)
    from dubai_planner.orchestration.crewai.board import CrewRunBoard

    board = CrewRunBoard(session_id="p", run_id="p", mode=probe.mode, today=TODAY,
                         user_message=WINTER)
    probe.factory.requirements_crew(board).kickoff(board)
    built = probe.factory.planning_crew(board)
    built.kickoff(board)
    out = {k: built.tasks[k].output.pydantic for k in RESEARCH_STAGES}  # type: ignore[union-attr]
    agents = AgentSuite(
        attractions=slow(out["attractions"]), accommodation=slow(out["accommodation"]),
        dining=slow(out["dining"]), transport=slow(out["transport"]),
        weather_events=slow(out["weather_events"]))
    with capture() as events:
        result = make_planner(agents=agents).run(WINTER, "s-par", today=TODAY)
    assert result.status is PlanStatus.APPROVED
    spans = [e for e in of_kind(events, "stage", TraceEventType.END)
             if e.agent in {STAGES[k].task for k in RESEARCH_STAGES}]
    assert len(spans) == 5
    starts = [s.started_at for s in spans]
    ends = [s.ended_at for s in spans if s.ended_at is not None]
    stage = (max(ends) - min(starts)).total_seconds()
    sequential = sum(s.duration_s or 0 for s in spans)
    assert sequential >= 5.0 and stage < 1.8, (stage, sequential)
    for a in spans:  # every pair of research spans overlaps in time
        for b in spans:
            assert a.started_at < (b.ended_at or b.started_at)
    # CrewAI's own task events show the same overlap
    crew = [e for e in of_kind(events, "crew_task", TraceEventType.END)
            if e.agent in {STAGES[k].task for k in RESEARCH_STAGES}]
    assert len(crew) == 5 and max(e.started_at for e in crew) < min(
        e.ended_at for e in crew if e.ended_at)


def test_hierarchical_research_runs_one_after_another(make_planner: Factory) -> None:
    """Documented limit: the manager's single executor cannot run tasks concurrently."""
    planner = make_planner(process="hierarchical")
    from dubai_planner.orchestration.crewai.board import CrewRunBoard

    board = CrewRunBoard(session_id="h", run_id="h", mode=planner.mode, today=TODAY,
                         user_message=WINTER)
    built = planner.factory.planning_crew(board)
    assert not any(t.async_execution for t in built.tasks.values())
    assert all(t.agent is None for t in built.tasks.values())


# ------------------------------------------------------------------ AC4.4


def test_ac44_midday_outdoor_slot_in_july_is_rejected_then_fixed(make_planner: Factory
                                                                 ) -> None:
    defect = HeatDefectComposer()
    with capture() as events:
        updates = list(make_planner(agents=AgentSuite(composer=defect)).stream(
            SUMMER, "s-heat", today=TODAY))
    result = updates[-1].result
    assert result is not None and result.status is PlanStatus.APPROVED
    assert len(defect.calls) == 2 and result.iterations == 1
    # the guardrail rejected attempt 1 for the heat rule, with feedback for the retry
    first, second = result.attempts
    assert "heat.midday_outdoor" in first.blocking_rules and not first.approved
    assert second.approved and second.iteration == 1
    assert any("heat.midday_outdoor" in f for f in defect.calls[1].feedback)
    assert defect.calls[1].locked_days == [1, 2]  # only day 1 (index 0) is re-planned
    assert result.itinerary is not None
    assert result.itinerary.days[1:] == defect.outputs[0].days[1:]
    # retry visible in the trace: our guardrail events and CrewAI's task retry count
    checks = [e for e in of_kind(events, "guardrail") if e.agent == "compose_itinerary"]
    assert [c.payload["passed"] for c in checks] == [False, True]
    assert checks[0].payload["rules"] == ["heat.midday_outdoor"]
    assert checks[0].payload["retry"] == 1 and checks[1].iteration == 1
    composed = [e for e in of_kind(events, "stage", TraceEventType.END)
                if e.agent == "compose_itinerary"]
    assert [e.iteration for e in composed] == [0, 1]
    done = [e for e in of_kind(events, "crew_task", TraceEventType.END)
            if e.agent == "compose_itinerary"]
    assert done[0].payload["retries"] == 1
    assert not [v for v in result.violations if v.severity is Severity.ERROR]


# ------------------------------------------------------------------ AC4.5


@pytest.mark.parametrize("process", ["sequential", "hierarchical"])
def test_ac45_process_selected_by_config_flag(process: str, make_planner: Factory,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_PROCESS", process)
    from dubai_planner.config import get_settings

    get_settings.cache_clear()
    planner = make_planner()  # no process argument: the setting decides
    assert planner.process == process
    result = planner.run(WINTER, f"s-{process}", today=TODAY)
    assert result.status is PlanStatus.APPROVED
    assert result.mode is PlannerMode(f"crewai_{process}")


def test_run_crewai_function_api(make_planner: Factory) -> None:
    planner = make_planner(process="hierarchical")
    updates = list(run_crewai(WINTER, "s-fn", planner=planner, today=TODAY))
    assert updates[-1].is_final and updates[-1].result is not None
    result = run_crewai("Make the budget AED 5500", "s-fn", planner=planner, stream=False,
                        today=TODAY)
    assert result.status is PlanStatus.APPROVED and result.saved_version == 2


# ------------------------------------------------------------------ AC4.6


def test_ac46_no_outbound_requests_during_a_crew_run(make_planner: Factory,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Every connect and DNS lookup is recorded; only loopback is let through.

    (On Windows, asyncio event loops use a 127.0.0.1 socket pair internally.)
    """
    from conftest import is_loopback

    attempts: list[Any] = []
    real_connect = socket.socket.connect

    def record(*args: Any, **_kwargs: Any) -> Any:
        attempts.append(args)
        raise OSError("network access is disabled in offline tests")

    def connect(sock: socket.socket, address: Any, *args: Any) -> Any:
        if is_loopback(address):
            return real_connect(sock, address, *args)
        return record(address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", record)
    monkeypatch.setattr(socket, "create_connection", record)
    monkeypatch.setattr(socket, "getaddrinfo", record)
    from crewai.events.listeners.tracing.utils import should_enable_tracing
    from crewai.telemetry.telemetry import Telemetry

    assert Telemetry()._is_telemetry_disabled()
    assert not should_enable_tracing()
    for process in ("sequential", "hierarchical"):
        result = make_planner(process=process).run(WINTER, f"s-tel-{process}", today=TODAY)
        assert result.status is PlanStatus.APPROVED
    assert attempts == []  # no DNS lookup, no socket connect: no telemetry, no traces


def test_crews_keep_no_replay_log_outside_the_project(make_planner: Factory) -> None:
    from dubai_planner.orchestration.crewai.board import CrewRunBoard
    from dubai_planner.orchestration.crewai.crew import _NoReplayLog

    planner = make_planner()
    board = CrewRunBoard(session_id="r", run_id="r", mode=planner.mode, today=TODAY,
                         user_message=WINTER)
    crew = planner.factory.planning_crew(board).crew
    assert isinstance(crew._task_output_handler, _NoReplayLog)
    assert crew.tracing is False and crew.cache is False


# ------------------------------------------------------------------ review loop


def test_rule_rejection_is_refined_once(make_planner: Factory) -> None:
    spy = ComposerSpy(inflate=1)
    result = make_planner(agents=AgentSuite(composer=spy)).run(WINTER, "s-loop", today=TODAY)
    assert result.status is PlanStatus.APPROVED
    assert result.iterations == 1 and len(spy.calls) == 2
    first, second = result.attempts
    assert "budget.total" in first.blocking_rules and second.approved
    assert any("budget.total" in f for f in spy.calls[1].feedback)
    assert spy.calls[1].locked_days == []  # a global issue unlocks every day


def test_loop_stops_at_cap_and_returns_best_unapproved(make_planner: Factory,
                                                       repo: Repository) -> None:
    spy = ComposerSpy(inflate=None)  # never fixes it
    with capture() as events:
        result = make_planner(agents=AgentSuite(composer=spy), max_review_iterations=2).run(
            WINTER, "s-cap", today=TODAY)
    assert result.status is PlanStatus.NEEDS_REVISION and not result.approved
    assert result.iterations == 2 and len(spy.calls) == 3 and len(result.attempts) == 3
    assert result.itinerary is not None and result.blocking_issues
    assert any(v.rule_id == "budget.total" for v in result.violations)
    checks = [e.payload for e in of_kind(events, "guardrail") if e.agent == "compose_itinerary"]
    assert [c["passed"] for c in checks] == [False, False, True] and checks[-1]["cap_reached"]
    record = repo.latest_itinerary("s-cap")
    assert record is not None and not record.approved  # saved, honestly unapproved
    run = repo.get_run(result.run_id)
    assert run is not None and run.status is RunStatus.SUCCEEDED


def test_zero_review_iterations_means_no_refinement(make_planner: Factory) -> None:
    spy = ComposerSpy(inflate=None)
    result = make_planner(agents=AgentSuite(composer=spy), max_review_iterations=0).run(
        WINTER, "s-zero", today=TODAY)
    assert result.status is PlanStatus.NEEDS_REVISION and len(spy.calls) == 1


def test_reviewer_rejection_runs_a_targeted_revision_crew(make_planner: Factory,
                                                          mock_llm: MockLLM) -> None:
    mock_llm.script("reviewer", json.dumps({
        "approved": False, "score": 0.4, "summary": "Day 2 is too rushed.",
        "issues": [{"severity": "error", "message": "Day 2 is too rushed", "day": 2,
                    "suggested_fix": "Drop one activity on day 2."}]}))
    spy = ComposerSpy()
    with capture() as events:
        result = make_planner(agents=AgentSuite(composer=spy)).run(WINTER, "s-day",
                                                                   today=TODAY)
    assert result.status is PlanStatus.APPROVED and result.iterations == 1
    second = spy.calls[1]
    assert second.locked_days == [0, 2] and second.previous == spy.outputs[0]
    assert any("Day 2 is too rushed" in f for f in second.feedback)
    before, after = spy.outputs
    assert after.days[0] == before.days[0] and after.days[2] == before.days[2]
    crews = [e for e in events if e.payload.get("kind") == "crew"
             and e.event is TraceEventType.START]
    assert [e.agent for e in crews] == ["requirements_crew", "planning_crew", "revision_crew"]
    assert crews[-1].iteration == 1


def test_reviewer_approval_never_overrides_hard_rule_errors(make_planner: Factory) -> None:
    def always_approve(data: Any, _llm: Any, _tools: Any) -> ReviewReport:
        return ReviewReport(approved=True, score=1.0, iteration=data.iteration)

    spy = ComposerSpy(inflate=None)
    result = make_planner(agents=AgentSuite(composer=spy, reviewer=always_approve)).run(
        WINTER, "s-safe", today=TODAY)
    assert result.status is PlanStatus.NEEDS_REVISION  # never silently approved
    assert result.review is not None and result.review.approved  # the reviewer said yes…
    assert any(v.severity is Severity.ERROR for v in result.violations)  # …the rules did not


# ------------------------------------------------------------------ clarification


def test_vague_request_asks_and_the_answer_completes_the_same_run(make_planner: Factory,
                                                                   repo: Repository) -> None:
    planner = make_planner()
    first = planner.run("plan a Dubai trip", "s-vague", today=TODAY)
    assert first.status is PlanStatus.NEEDS_CLARIFICATION and first.clarification_question
    assert first.missing and first.itinerary is None and first.saved_version is None
    assert planner.pending_question("s-vague") == first.clarification_question
    run = repo.get_run(first.run_id)
    assert run is not None and run.status is RunStatus.INTERRUPTED
    second = planner.run("5 days in Feb, AED 8000", "s-vague", today=TODAY)
    assert second.status is PlanStatus.APPROVED and second.run_id == first.run_id
    assert second.request is not None and second.request.num_days == 5
    assert second.request.total_budget_aed() == Decimal("8000")
    assert planner.pending_question("s-vague") is None
    run = repo.get_run(first.run_id)
    assert run is not None and run.status is RunStatus.SUCCEEDED
    roles = [m.role for m in repo.chat_history("s-vague")]
    assert roles == [ChatRole.USER, ChatRole.ASSISTANT, ChatRole.USER, ChatRole.ASSISTANT]


def test_clarification_rounds_are_capped(make_planner: Factory) -> None:
    planner = make_planner()
    for _ in range(3):
        result = planner.run("plan a Dubai trip", "s-cap-q", today=TODAY)
        assert result.status is PlanStatus.NEEDS_CLARIFICATION
    assert planner.pending_question("s-cap-q") is None  # gave up: the next message is new
    fresh = planner.run(WINTER, "s-cap-q", today=TODAY)
    assert fresh.status is PlanStatus.APPROVED and fresh.run_id != result.run_id


# ------------------------------------------------------------------ sessions


def test_refinement_builds_on_the_session_draft(make_planner: Factory,
                                                repo: Repository) -> None:
    planner = make_planner()
    first = planner.run(WINTER, "s-refine", today=TODAY)
    second = planner.run("Change the budget to AED 5000", "s-refine", today=TODAY)
    assert first.request is not None and second.request is not None
    assert second.request.start_date == first.request.start_date  # dates carried over
    assert second.request.total_budget_aed() == Decimal("5000")
    assert second.saved_version == 2
    versions = repo.list_itineraries("s-refine")
    assert [v.version for v in versions] == [1, 2] and versions[1].parent_id == versions[0].id


def test_concurrent_sessions_are_isolated(make_planner: Factory, repo: Repository) -> None:
    planner = make_planner()
    results: dict[str, Any] = {}
    requests = {"s-a": WINTER, "s-b": "3 adults, 1-3 March 2027, AED 9000, shopping"}
    with capture() as events:
        threads = [threading.Thread(target=lambda s=s, m=m: results.__setitem__(
            s, planner.run(m, s, today=TODAY))) for s, m in requests.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    a, b = results["s-a"], results["s-b"]
    assert a.status is PlanStatus.APPROVED and b.status is PlanStatus.APPROVED
    assert a.run_id != b.run_id
    assert a.request is not None and b.request is not None
    assert a.request.travellers.adults == 2 and b.request.travellers.adults == 3
    assert {e.run_id for e in events} == {a.run_id, b.run_id}
    for session, result in results.items():
        record = repo.latest_itinerary(session)
        assert record is not None and record.run_id == result.run_id and record.version == 1


# ------------------------------------------------------------------ errors


def test_agent_failure_fails_the_run_loudly_and_once(make_planner: Factory,
                                                     repo: Repository) -> None:
    calls: list[int] = []

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        calls.append(1)
        raise StructuredOutputError("dining", "Selection", ["bad json"], "{")

    result = make_planner(agents=AgentSuite(dining=broken)).run(WINTER, "s-err", today=TODAY)
    assert result.status is PlanStatus.FAILED and not result.approved
    assert result.error is not None and "StructuredOutputError" in result.error
    assert len(calls) == 1  # CrewAI re-tries a raising tool; the work is not repeated
    assert result.itinerary is None and repo.latest_itinerary("s-err") is None
    run = repo.get_run(result.run_id)
    assert run is not None and run.status is RunStatus.FAILED and run.error == result.error


def test_failed_run_can_be_resumed(make_planner: Factory, repo: Repository) -> None:
    state = {"fail": True}

    def flaky_dining(data: Any, llm: Any, tools: Toolbox) -> Any:
        if state["fail"]:
            raise StructuredOutputError("dining", "Selection", ["bad json"], "{")
        return dining.run(data, llm, tools)

    planner = make_planner(agents=AgentSuite(dining=flaky_dining))
    failed = planner.run(WINTER, "s-resume", today=TODAY)
    assert failed.status is PlanStatus.FAILED
    state["fail"] = False
    resumed = planner.resume("s-resume")
    assert resumed.status is PlanStatus.APPROVED and resumed.run_id == failed.run_id
    run = repo.get_run(failed.run_id)
    assert run is not None and run.status is RunStatus.SUCCEEDED
    assert [m.role for m in repo.chat_history("s-resume")].count(ChatRole.USER) == 1
    with pytest.raises(ValueError, match="no unfinished run"):
        planner.resume("s-resume")


def test_transient_provider_error_is_retried(make_planner: Factory) -> None:
    calls: list[int] = []

    def rate_limited_once(data: Any, llm: Any, tools: Toolbox) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise LLMCallError("groq: RateLimitError: Error code: 429")
        return attractions.run(data, llm, tools)

    with capture() as events:
        result = make_planner(agents=AgentSuite(attractions=rate_limited_once)).run(
            WINTER, "s-429", today=TODAY)
    assert result.status is PlanStatus.APPROVED and len(calls) == 2
    retries = of_kind(events, "retry")
    assert len(retries) == 1 and retries[0].agent == "research_attractions"


def test_missing_api_key_fails_with_a_clear_error(make_planner: Factory, toolbox: Toolbox,
                                                  repo: Repository,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from dubai_planner.config import Settings

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    no_key = Settings(_env_file=None, llm_provider="groq")  # type: ignore[call-arg]
    result = CrewAIPlanner(tools=toolbox, repo=repo, settings=no_key).run(
        WINTER, "s-key", today=TODAY)
    assert result.status is PlanStatus.FAILED
    assert result.error is not None and "GROQ_API_KEY" in result.error


def test_secret_key_never_leaks_into_results_runs_or_traces(
        toolbox: Toolbox, repo: Repository, monkeypatch: pytest.MonkeyPatch) -> None:
    canary = "gsk_CANARY_do_not_leak_0123456789"
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", canary)
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    from dubai_planner.config import get_settings

    get_settings.cache_clear()
    with capture() as events:
        result = CrewAIPlanner(tools=toolbox, repo=repo, retry=FAST_RETRY).run(
            WINTER, "s-canary", today=TODAY)
    assert result.status is PlanStatus.FAILED  # sockets are blocked: the provider is down
    run = repo.get_run(result.run_id)
    texts = [result.model_dump_json(), run.model_dump_json() if run else "",
             *(e.model_dump_json() for e in events),
             *(m.model_dump_json() for m in repo.chat_history("s-canary"))]
    assert not any(canary in t for t in texts)


def test_cancelled_stream_stops_the_crew_and_saves_nothing(make_planner: Factory,
                                                           repo: Repository) -> None:
    gate = threading.Event()

    def slow_budget(data: Any, llm: Any, tools: Toolbox) -> Any:
        from dubai_planner.agents import budget

        gate.wait(5)
        return budget.run(data, llm, tools)

    planner = make_planner(agents=AgentSuite(budget=slow_budget))
    stream = planner.stream(WINTER, "s-cancel", today=TODAY)
    first = next(stream)
    assert first.node == "parse_request"
    gate.set()
    stream.close()  # the caller walks away mid-run
    assert repo.latest_itinerary("s-cancel") is None
    runs = repo.list_runs("s-cancel")
    assert len(runs) == 1 and runs[0].status is RunStatus.INTERRUPTED


def test_message_validation() -> None:
    planner = CrewAIPlanner(persist=False, llm=MockLLM())
    with pytest.raises(ValueError, match="characters"):
        planner.run("   ", "s")
    with pytest.raises(ValueError, match="session_id"):
        planner.run(WINTER, " ")
    with pytest.raises(ValueError, match="unknown CrewAI process"):
        CrewAIPlanner(process="parallel", persist=False)  # type: ignore[arg-type]



# ------------------------------------------------------------------ real model


@pytest.mark.llm
def test_crewai_real_model_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One sequential crew run on the configured provider (not an AC4 criterion).

    Proves CrewAI 1.15 drives the configured model (Groq ``openai/gpt-oss-20b``: native
    function calling, output limits) end to end. One scenario, data sources offline;
    a smoke test, not a quality evaluation (that is Phase 7).
    """
    import time

    from dubai_planner.config import get_settings

    if get_settings().llm_provider.value == "mock":
        pytest.skip("set LLM_PROVIDER to a real provider (shell or .env)")
    monkeypatch.setenv("HTTP_OFFLINE", "true")
    get_settings.cache_clear()
    repo = Repository(tmp_path / "app.db")
    planner = CrewAIPlanner(process="sequential", tools=Toolbox.default(), repo=repo)
    started = time.monotonic()
    with capture() as events:
        result = planner.run(WINTER, "s-real", today=TODAY)
    elapsed = time.monotonic() - started
    run = repo.get_run(result.run_id)
    crew_calls = of_kind(events, "crew_agent")
    print(f"\nCrewAI real run: status={result.status.value} in {elapsed:.1f}s, "
          f"iterations={result.iterations}, run llm_calls={run.llm_calls if run else '?'}, "
          f"tokens={run.tokens if run else '?'}, crew-agent calls={len(crew_calls)} "
          f"({sum(e.tokens_in + e.tokens_out for e in crew_calls)} tokens), "
          f"error={result.error}")
    assert result.status in (PlanStatus.APPROVED, PlanStatus.NEEDS_REVISION), result.error
    assert elapsed < 300
    assert result.itinerary is not None and result.request is not None
    assert Itinerary.model_validate_json(result.itinerary.model_dump_json())
    assert {(e.agent, e.payload["tool"]) for e in of_kind(events, "tool")} >= set(
        ROLE_TOOLS.items())
    assert crew_calls and all(e.tokens_in > 0 for e in crew_calls)
