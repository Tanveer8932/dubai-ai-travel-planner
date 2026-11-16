"""Phase 3: LangGraph orchestration end to end (offline, mock LLM unless marked ``llm``).

Mock-provider results here prove orchestration behaviour (routing, parallelism, loop
bounds, checkpointing), never real-model quality; AC3.6 is the only real-model test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command, RetryPolicy

from dubai_planner.agents import (
    accommodation,
    attractions,
    composer,
    dining,
    transport,
    weather_events,
)
from dubai_planner.llm.errors import LLMCallError, StructuredOutputError
from dubai_planner.llm.mock import MockLLM
from dubai_planner.models import (
    ChatRole,
    Itinerary,
    PlanStatus,
    ReviewReport,
    RunStatus,
    Severity,
    TraceEvent,
    TraceEventType,
)
from dubai_planner.orchestration.langgraph import AgentSuite, LangGraphPlanner, is_transient
from dubai_planner.orchestration.langgraph import nodes as n
from dubai_planner.orchestration.langgraph.state import new_run_input
from dubai_planner.storage import Repository
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import capture

TODAY = date(2026, 11, 16)
WINTER = "2 adults, 10-12 Feb 2027, AED 6000, history and food"
FAST_RETRY = RetryPolicy(max_attempts=2, initial_interval=0.01, backoff_factor=1.0,
                         max_interval=0.01, jitter=False, retry_on=is_transient)
WORKER = Path(__file__).with_name("langgraph_worker.py")

Factory = Callable[..., LangGraphPlanner]


@pytest.fixture
def make_planner(tmp_path: Path, toolbox: Toolbox, mock_llm: MockLLM) -> Iterator[Factory]:
    """Planners sharing one checkpoint DB and one app DB (like app restarts)."""
    stack = ExitStack()
    repo = Repository(tmp_path / "app.db")

    def make(**kwargs: Any) -> LangGraphPlanner:
        kwargs.setdefault("llm", mock_llm)
        kwargs.setdefault("tools", toolbox)
        kwargs.setdefault("repo", repo)
        kwargs.setdefault("retry", FAST_RETRY)
        return stack.enter_context(LangGraphPlanner.open(tmp_path / "cp.db", **kwargs))

    yield make
    stack.close()


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "app.db")


def node_events(events: list[TraceEvent], name: str, kind: TraceEventType) -> list[TraceEvent]:
    return [e for e in events
            if e.agent == name and e.event is kind and e.payload.get("kind") == "node"]


class ComposerSpy:
    """Real composer; the first ``inflate`` attempts (None = all) come back over budget."""

    def __init__(self, inflate: int | None = 0) -> None:
        self.inflate = inflate
        self.calls: list[composer.ComposerInput] = []
        self.outputs: list[Itinerary] = []

    def __call__(self, data: composer.ComposerInput, llm: Any, tools: Toolbox) -> Itinerary:
        self.calls.append(data)
        itinerary = composer.run(data, llm, tools)
        if self.inflate is None or len(self.calls) <= self.inflate:
            total = data.request.total_budget_aed() * 2
            itinerary = itinerary.model_copy(update={
                "cost": itinerary.cost.model_copy(update={"total": total})})
        self.outputs.append(itinerary)
        return itinerary


# ------------------------------------------------------------------ AC3.1


def test_ac31_end_to_end_mock_run_is_approved_and_persisted(make_planner: Factory,
                                                            repo: Repository) -> None:
    planner = make_planner()
    with capture() as events:
        updates = list(planner.stream(WINTER, "s-e2e", today=TODAY))
    result = updates[-1].result
    assert result is not None and result.status is PlanStatus.APPROVED and result.approved
    itinerary = result.itinerary
    assert itinerary is not None and result.request is not None
    assert Itinerary.model_validate_json(itinerary.model_dump_json()) == itinerary
    assert itinerary.cost.total <= result.request.total_budget_aed() * Decimal("1.05")
    assert not [v for v in result.violations if v.severity is Severity.ERROR]
    assert itinerary.venue_ids() <= (planner.snapshot("s-e2e").values["research"].known_ids())
    # streamed progress: one update per node, then the final one
    nodes = [u.node for u in updates]
    assert nodes[0] == n.PARSE and nodes[-1] == "__end__"
    assert set(n.RESEARCH_NODES) <= set(nodes) and n.FINALIZE in nodes
    # persisted: session, messages, approved v1, run record
    record = repo.latest_itinerary("s-e2e")
    assert record is not None and record.version == 1 and record.approved
    assert record.run_id == result.run_id and result.saved_version == 1
    assert [m.role for m in repo.chat_history("s-e2e")] == [ChatRole.USER, ChatRole.ASSISTANT]
    run = repo.get_run(result.run_id)
    assert run is not None and run.status is RunStatus.SUCCEEDED and run.llm_calls >= 6
    # every event belongs to this run, in LangGraph mode
    assert events and all(e.run_id == result.run_id and e.mode == "langgraph" for e in events)


# ------------------------------------------------------------------ AC3.2


def test_ac32_research_nodes_run_in_parallel(make_planner: Factory, mock_llm: MockLLM,
                                             toolbox: Toolbox) -> None:
    """Each research node stubbed to sleep 1 s: stage < 1.8 s (vs ≈ 5 s sequential)."""
    probe = make_planner()
    probe.run(WINTER, "warm", today=TODAY)  # real outputs to return from the stubs
    research = probe.snapshot("warm").values["research"]

    def slow(output: Any) -> Callable[..., Any]:
        def stub(*_args: Any, **_kwargs: Any) -> Any:
            time.sleep(1.0)
            return output
        return stub

    agents = AgentSuite(
        attractions=slow(attractions.AttractionsResult(options=research.attractions)),
        accommodation=slow(accommodation.AccommodationResult(options=research.hotels)),
        dining=slow(dining.DiningResult(options=research.dining)),
        transport=slow(research.transport),
        weather_events=slow(weather_events.WeatherEventsResult(
            weather=research.weather, events=research.events)))
    planner = make_planner(agents=agents)
    with capture() as events:
        result = planner.run(WINTER, "s-par", today=TODAY)
    assert result.status is PlanStatus.APPROVED
    spans = [node_events(events, name, TraceEventType.END)[0] for name in n.RESEARCH_NODES]
    starts = [s.started_at for s in spans]
    ends = [s.ended_at for s in spans if s.ended_at is not None]
    stage = (max(ends) - min(starts)).total_seconds()
    sequential = sum(s.duration_s or 0 for s in spans)
    assert sequential >= 5.0 and stage < 1.8, (stage, sequential)
    # every pair of research spans overlaps in time
    for a in spans:
        for b in spans:
            assert a.started_at < (b.ended_at or b.started_at)
    assert transport.AGENT  # the transport stub replaced the deterministic agent


# ------------------------------------------------------------------ AC3.3 review loop


def test_ac33_over_budget_first_attempt_is_refined_once(make_planner: Factory) -> None:
    spy = ComposerSpy(inflate=1)
    planner = make_planner(agents=AgentSuite(composer=spy))
    with capture() as events:
        updates = list(planner.stream(WINTER, "s-loop", today=TODAY))
    result = updates[-1].result
    assert result is not None and result.status is PlanStatus.APPROVED
    assert result.iterations == 1 and len(spy.calls) == 2
    assert [u.node for u in updates].count(n.REFINE) == 1
    first, second = result.attempts
    assert "budget.total" in first.blocking_rules and not first.approved
    assert second.approved and second.iteration == 1
    # targeted: the composer got the reviewer's feedback; a global issue unlocks all days
    assert any("budget.total" in f for f in spy.calls[1].feedback)
    assert spy.calls[1].locked_days == []
    # trace: every event carries the run id; the second compose runs in iteration 1
    assert all(e.run_id == result.run_id for e in events)
    assert [e.iteration for e in node_events(events, n.COMPOSE, TraceEventType.END)] == [0, 1]


def test_ac33_loop_stops_at_cap_and_returns_best_unapproved(make_planner: Factory,
                                                            repo: Repository) -> None:
    spy = ComposerSpy(inflate=None)  # never fixes it
    planner = make_planner(agents=AgentSuite(composer=spy), max_review_iterations=2)
    result = planner.run(WINTER, "s-cap", today=TODAY)
    assert result.status is PlanStatus.NEEDS_REVISION and not result.approved
    assert result.iterations == 2 and len(spy.calls) == 3 and len(result.attempts) == 3
    assert result.itinerary is not None and result.blocking_issues
    assert any(v.rule_id == "budget.total" for v in result.violations)
    record = repo.latest_itinerary("s-cap")
    assert record is not None and not record.approved  # saved, honestly unapproved
    run = repo.get_run(result.run_id)
    assert run is not None and run.status is RunStatus.SUCCEEDED


def test_zero_review_iterations_means_no_refinement(make_planner: Factory) -> None:
    spy = ComposerSpy(inflate=None)
    result = make_planner(agents=AgentSuite(composer=spy), max_review_iterations=0).run(
        WINTER, "s-zero", today=TODAY)
    assert result.status is PlanStatus.NEEDS_REVISION and len(spy.calls) == 1


def test_targeted_refinement_replans_only_the_faulty_day(make_planner: Factory,
                                                         mock_llm: MockLLM) -> None:
    mock_llm.script("reviewer", json.dumps({
        "approved": False, "score": 0.4, "summary": "Day 2 is too rushed.",
        "issues": [{"severity": "error", "message": "Day 2 is too rushed", "day": 2,
                    "suggested_fix": "Drop one activity on day 2."}]}))
    spy = ComposerSpy()
    result = make_planner(agents=AgentSuite(composer=spy)).run(WINTER, "s-day", today=TODAY)
    assert result.status is PlanStatus.APPROVED and result.iterations == 1
    second = spy.calls[1]
    assert second.locked_days == [0, 2] and second.previous == spy.outputs[0]
    assert any("Day 2 is too rushed" in f for f in second.feedback)
    before, after = spy.outputs
    assert after.days[0] == before.days[0] and after.days[2] == before.days[2]


def test_reviewer_approval_never_overrides_hard_rule_errors(make_planner: Factory) -> None:
    def always_approve(data: Any, _llm: Any, _tools: Any) -> ReviewReport:
        return ReviewReport(approved=True, score=1.0, iteration=data.iteration)

    spy = ComposerSpy(inflate=None)
    result = make_planner(agents=AgentSuite(composer=spy, reviewer=always_approve)).run(
        WINTER, "s-safe", today=TODAY)
    assert result.status is PlanStatus.NEEDS_REVISION  # never silently approved
    assert any(v.severity is Severity.ERROR for v in result.violations)


# ------------------------------------------------------------------ AC3.4 clarification


def test_ac34_vague_request_interrupts_and_command_resume_completes(make_planner: Factory
                                                                    ) -> None:
    graph = make_planner().graph
    cfg = {"configurable": {"thread_id": "s-raw"}}
    msg = "plan a Dubai trip"
    from dubai_planner.models import ChatMessage

    graph.invoke(new_run_input(session_id="s-raw", run_id="r-raw", message=msg, today=TODAY,
                               user_message=ChatMessage(role=ChatRole.USER, content=msg)), cfg)
    snap = graph.get_state(cfg)
    assert snap.next == (n.CLARIFY,) and snap.interrupts
    assert "budget" in snap.interrupts[0].value["question"]
    final = graph.invoke(Command(resume="5 days in Feb, AED 8000"), cfg)
    assert final["status"] is PlanStatus.APPROVED
    assert final["request"].num_days == 5 and final["request"].budget_aed == Decimal("8000.00")


def test_ac34_service_answers_pending_question(make_planner: Factory, repo: Repository) -> None:
    planner = make_planner()
    first = planner.run("plan a Dubai trip", "s-clar", today=TODAY)
    assert first.status is PlanStatus.NEEDS_CLARIFICATION and first.clarification_question
    assert {"start_date", "budget_aed"} <= set(first.missing)
    assert planner.pending_question("s-clar") == first.clarification_question
    run = repo.get_run(first.run_id)
    assert run is not None and run.status is RunStatus.INTERRUPTED
    second = planner.run("5 days in Feb, AED 8000", "s-clar", today=TODAY)
    assert second.status is PlanStatus.APPROVED and second.run_id == first.run_id
    assert second.request is not None and second.request.num_days == 5
    roles = [m.role for m in repo.chat_history("s-clar")]
    assert roles == [ChatRole.USER, ChatRole.ASSISTANT, ChatRole.USER, ChatRole.ASSISTANT]


def test_clarification_rounds_are_capped(make_planner: Factory) -> None:
    planner = make_planner()
    planner.run("plan a Dubai trip", "s-vague", today=TODAY)
    for answer in ("hmm", "not sure yet"):
        assert planner.run(answer, "s-vague", today=TODAY).status \
            is PlanStatus.NEEDS_CLARIFICATION
    last = planner.run("no idea", "s-vague", today=TODAY)  # third answer: stop asking
    assert last.status is PlanStatus.NEEDS_CLARIFICATION and last.itinerary is None
    assert planner.pending_question("s-vague") is None and not planner.snapshot("s-vague").next


# ------------------------------------------------------------------ sessions


def test_sessions_are_isolated(make_planner: Factory, repo: Repository) -> None:
    planner = make_planner()
    a = planner.run(WINTER, "s-a", today=TODAY)
    b = planner.run("plan a Dubai trip", "s-b", today=TODAY)
    assert a.status is PlanStatus.APPROVED
    assert b.status is PlanStatus.NEEDS_CLARIFICATION  # nothing leaked from session A
    assert b.draft is not None and b.draft.budget_aed is None
    again = planner.result("s-a")
    assert again is not None and again.itinerary == a.itinerary  # A is untouched
    assert [r.session_id for r in repo.list_itineraries("s-a")] == ["s-a"]
    assert repo.list_itineraries("s-b") == []


def test_follow_up_is_a_new_version_with_context(make_planner: Factory,
                                                 repo: Repository) -> None:
    planner = make_planner()
    v1 = planner.run(WINTER, "s-ver", today=TODAY)
    v2 = planner.run("make the budget AED 9000", "s-ver", today=TODAY)
    assert v1.status is v2.status is PlanStatus.APPROVED and v1.run_id != v2.run_id
    assert v2.request is not None and v2.request.budget_aed == Decimal("9000.00")
    assert v2.request.start_date == date(2027, 2, 10)  # dates carried over from v1
    records = repo.list_itineraries("s-ver")
    assert [r.version for r in records] == [1, 2] and records[1].parent_id == records[0].id
    assert v2.saved_version == 2 and v2.attempts[0].iteration == 0  # loop state was reset


def test_invalid_input_is_rejected(make_planner: Factory) -> None:
    planner = make_planner()
    with pytest.raises(ValueError, match="1-4000"):
        planner.run("   ", "s-bad")
    with pytest.raises(ValueError, match="1-4000"):
        planner.run("x" * 4001, "s-bad")
    with pytest.raises(ValueError, match="no unfinished run"):
        planner.resume("s-bad")
    with pytest.raises(ValueError, match="session_id"):
        planner.run("hi", " ")


# ------------------------------------------------------------------ AC3.5 restart


def test_ac35_killed_process_resumes_from_last_checkpoint(tmp_path: Path) -> None:
    cp, db = str(tmp_path / "cp.db"), str(tmp_path / "app.db")
    env = {**os.environ, "LLM_PROVIDER": "mock", "HTTP_OFFLINE": "true"}

    def worker(mode: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - our own worker script, fixed arguments
            [sys.executable, str(WORKER), mode, cp, db, "s-crash"],
            capture_output=True, text=True, env=env, timeout=240, check=False)

    crashed = worker("crash")
    assert crashed.returncode == 17, crashed.stderr[-2000:]  # killed inside compose
    resumed = worker("resume")
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    out = json.loads(resumed.stdout.strip().splitlines()[-1])
    assert out["status"] == "approved" and out["version"] == 1
    # the new process continued at compose: parse/research/budget were not re-run
    assert out["nodes"][0] == n.COMPOSE
    assert not set(out["nodes"]) & {n.PARSE, n.MERGE, n.BUDGET, *n.RESEARCH_NODES}
    run = Repository(db).get_run(out["run_id"])
    assert run is not None and run.status is RunStatus.SUCCEEDED


# ------------------------------------------------------------------ errors / cancellation


def test_failed_branch_is_reported_and_resume_reruns_only_it(make_planner: Factory,
                                                             repo: Repository) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise StructuredOutputError("dining", "Selection", ["not JSON"], "oops")

    failed = make_planner(agents=AgentSuite(dining=broken)).run(WINTER, "s-err", today=TODAY)
    assert failed.status is PlanStatus.FAILED and failed.itinerary is None
    assert failed.error is not None and "StructuredOutputError" in failed.error
    run = repo.get_run(failed.run_id)
    assert run is not None and run.status is RunStatus.FAILED and run.error == failed.error

    fixed = make_planner()  # e.g. after the provider recovers or the app restarts
    with capture() as events:
        resumed = fixed.resume("s-err")
    assert resumed.status is PlanStatus.APPROVED and resumed.run_id == failed.run_id
    started = {e.agent for e in events
               if e.event is TraceEventType.START and e.payload.get("kind") == "node"}
    # the four branches that succeeded were checkpointed and are not re-run
    assert n.RESEARCH_DINING in started and n.PARSE not in started
    assert not started & {n.RESEARCH_ATTRACTIONS, n.RESEARCH_ACCOMMODATION,
                          n.RESEARCH_TRANSPORT, n.RESEARCH_WEATHER}


def test_transient_llm_error_is_retried(make_planner: Factory) -> None:
    calls = {"n": 0}

    def flaky(data: Any, llm: Any, tools: Toolbox) -> Itinerary:
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMCallError("groq: RateLimitError: Error code: 429 - rate limit reached")
        return composer.run(data, llm, tools)

    result = make_planner(agents=AgentSuite(composer=flaky)).run(WINTER, "s-429", today=TODAY)
    assert result.status is PlanStatus.APPROVED and calls["n"] == 2


def test_permanent_llm_error_fails_without_retry(make_planner: Factory) -> None:
    calls = {"n": 0}

    def unauthorised(*_args: Any, **_kwargs: Any) -> Any:
        calls["n"] += 1
        raise LLMCallError("groq: AuthenticationError: Error code: 401 - invalid api key")

    result = make_planner(agents=AgentSuite(composer=unauthorised)).run(
        WINTER, "s-401", today=TODAY)
    assert result.status is PlanStatus.FAILED and calls["n"] == 1
    assert result.error is not None and "401" in result.error


def test_cancelled_stream_is_recorded_and_resumable(make_planner: Factory,
                                                    repo: Repository) -> None:
    planner = make_planner()
    stream = planner.stream(WINTER, "s-cancel", today=TODAY)
    seen = []
    for update in stream:
        seen.append(update.node)
        if update.node == n.MERGE:
            break
    stream.close()  # e.g. the user closed the page
    state = planner.result("s-cancel")
    assert state is not None and state.status is PlanStatus.CANCELLED
    run = repo.get_run(state.run_id)
    assert run is not None and run.status is RunStatus.INTERRUPTED
    resumed = planner.resume("s-cancel")
    assert resumed.status is PlanStatus.APPROVED and resumed.run_id == state.run_id
    assert n.FINALIZE not in seen


def test_missing_api_key_fails_with_a_clear_error(make_planner: Factory,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from dubai_planner.config import Settings
    from dubai_planner.llm import factory

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    no_key = Settings(_env_file=None, llm_provider="groq")  # type: ignore[call-arg]
    monkeypatch.setattr(factory, "get_settings", lambda: no_key)  # ignore the real .env
    result = make_planner(llm=None).run(WINTER, "s-key", today=TODAY)
    assert result.status is PlanStatus.FAILED
    assert result.error is not None and "GROQ_API_KEY" in result.error


def test_api_key_never_reaches_traces_results_or_db(make_planner: Factory, repo: Repository,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langchain_groq")
    from dubai_planner.config import get_settings

    canary = "gsk_" + "canary" * 5  # fake value; must never show up anywhere
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", canary)
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    get_settings.cache_clear()
    with capture() as events:
        result = make_planner(llm=None).run(WINTER, "s-secret", today=TODAY)
    assert result.status is PlanStatus.FAILED  # sockets are blocked: the call cannot succeed
    dumped = [e.model_dump_json() for e in events] + [result.model_dump_json()]
    run = repo.get_run(result.run_id)
    dumped += [run.model_dump_json() if run else ""]
    dumped += [m.model_dump_json() for m in repo.chat_history("s-secret")]
    assert all(canary not in text for text in dumped)


# ------------------------------------------------------------------ AC3.6 (real model)


@pytest.mark.llm
def test_ac36_real_llm_smoke_run(tmp_path: Path, toolbox: Toolbox) -> None:
    """One scenario with the configured real provider in < 5 min (LLM_PROVIDER from .env)."""
    from dubai_planner.config import LLMProvider, get_settings

    settings = get_settings()
    if settings.llm_provider is LLMProvider.MOCK:
        pytest.skip("set LLM_PROVIDER to a real provider (ollama/groq/gemini) for AC3.6")
    with LangGraphPlanner.open(tmp_path / "cp.db", tools=toolbox,
                               repo=Repository(tmp_path / "app.db")) as planner, \
            capture() as events:
        started = time.perf_counter()
        result = planner.run(WINTER, "s-smoke", today=TODAY)
        elapsed = time.perf_counter() - started
        research = planner.snapshot("s-smoke").values.get("research")
    calls = [e for e in events if e.event is TraceEventType.LLM_CALL]
    print(json.dumps({  # evidence for implementationPlan.md (no secrets in here)
        "provider": settings.llm_provider.value, "model": settings.model_for(None),
        "status": result.status.value, "seconds": round(elapsed, 1),
        "iterations": result.iterations, "llm_calls": len(calls),
        "tokens": sum(e.tokens_in + e.tokens_out for e in calls),
        "attempts": [a.model_dump(mode="json") for a in result.attempts],
        "error": result.error}))
    assert elapsed < 300, elapsed
    assert result.status in (PlanStatus.APPROVED, PlanStatus.NEEDS_REVISION), result.error
    assert result.itinerary is not None and result.request is not None
    assert research is not None and result.itinerary.venue_ids() <= research.known_ids()
