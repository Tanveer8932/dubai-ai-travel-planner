"""Phase 6 acceptance criteria on real orchestration runs (mock LLM, offline).

The traces analysed here are the ones both planners *persist* while they run
(``RunStats`` → ``trace_events``); the tests read them back from SQLite only, as the
Execution page does. AC6.1-AC6.5, plus failed / cancelled runs, comparison and secrets.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from planner_helpers import TODAY, WINTER, ComposerSpy, HeatDefectComposer, slow

from dubai_planner.agents import accommodation, attractions, composer, dining, weather_events
from dubai_planner.agents.suite import AgentSuite
from dubai_planner.config import get_settings
from dubai_planner.llm.mock import MockLLM
from dubai_planner.models import Itinerary, PlannerMode, PlanStatus, RunStatus, TraceEventType
from dubai_planner.orchestration.crewai.runner import CrewAIPlanner
from dubai_planner.orchestration.langgraph.graph import describe_graph
from dubai_planner.orchestration.langgraph.runner import LangGraphPlanner
from dubai_planner.services import observability as ob
from dubai_planner.services.planner import open_planner
from dubai_planner.storage import Repository
from dubai_planner.tools.toolbox import Toolbox

SUMMER = "2 adults, 14-16 July 2027, AED 9000, beach and culture"
CANARY = "gsk_CANARY_phase6_trace_never_shows_this_99"
Make = Callable[..., Any]


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "app.db")


@pytest.fixture
def make(tmp_path: Path, toolbox: Toolbox, repo: Repository) -> Iterator[Make]:
    stack = ExitStack()

    def factory(mode: PlannerMode, **kwargs: Any) -> Any:
        kwargs.setdefault("llm", MockLLM())
        return stack.enter_context(open_planner(
            mode, checkpoint_path=tmp_path / f"cp-{mode.value}-{len(stack._exit_callbacks)}.db",
            tools=toolbox, repo=repo, **kwargs))

    yield factory
    stack.close()


@pytest.fixture
def slow_research(make: Make) -> Callable[[float], AgentSuite]:
    """Research agents that sleep, returning real outputs from a probe run."""
    probe = make(PlannerMode.LANGGRAPH)
    probe.run(WINTER, "probe", today=TODAY)
    research = probe.snapshot("probe").values["research"]

    def build(seconds: float, **extra: Any) -> AgentSuite:
        return AgentSuite(
            attractions=slow(attractions.AttractionsResult(options=research.attractions),
                             seconds),
            accommodation=slow(accommodation.AccommodationResult(options=research.hotels),
                               seconds),
            dining=slow(dining.DiningResult(options=research.dining), seconds),
            transport=slow(research.transport, seconds),
            weather_events=slow(weather_events.WeatherEventsResult(
                weather=research.weather, events=research.events), seconds), **extra)

    return build


def stored(repo: Repository, run_id: str) -> ob.RunTrace:
    trace = ob.load_trace(repo, run_id)
    assert trace is not None and trace.events, "no persisted trace"
    return trace


def lane_overlap(rows: list[ob.TimelineRow], names: set[str], kind: str) -> tuple[int, int]:
    chosen = [r for r in rows if r.name in names and r.kind == kind]
    pairs = [(a, b) for i, a in enumerate(chosen) for b in chosen[i + 1:]]
    return sum(a.start_s < b.end_s and b.start_s < a.end_s for a, b in pairs), len(pairs)


def figure_parts(trace: ob.RunTrace, monkeypatch: pytest.MonkeyPatch) -> Any:
    app = Path(__file__).resolve().parents[2] / "app"
    monkeypatch.syspath_prepend(str(app))
    import ui_execution

    return ui_execution.timeline_figure(trace)


# ------------------------------------------------------------------ AC6.1


def test_ac61_langgraph_gantt_shows_parallel_research_and_iteration_bands(
        make: Make, slow_research: Callable[..., AgentSuite], repo: Repository,
        monkeypatch: pytest.MonkeyPatch) -> None:
    agents = slow_research(0.4, composer=ComposerSpy(inflate=1))
    result = make(PlannerMode.LANGGRAPH, agents=agents).run(WINTER, "s-61", today=TODAY)
    assert result.status is PlanStatus.APPROVED and result.iterations == 1
    trace = stored(repo, result.run_id)
    rows = ob.timeline_rows(trace)
    research = set(ob.RESEARCH_STEPS)
    assert {r.lane for r in rows} >= research  # one lane per node
    overlapping, pairs = lane_overlap(rows, research, "node")
    assert pairs == 10 and overlapping == 10  # all five research nodes overlap in time
    par = ob.parallelism(ob.step_spans(trace))
    assert par is not None and par.parallel and par.window_s < 1.0 < par.sum_s
    bands = ob.iteration_bands(rows)
    assert [b.iteration for b in bands] == [0, 1]
    assert bands[0].end_s <= bands[1].start_s  # separate bands, in order
    fig = figure_parts(trace, monkeypatch)
    names = [t.name for t in fig.data]
    assert "steps · first draft" in names and "steps · revision 1" in names
    labels = [a.text for a in fig.layout.annotations]
    assert labels == ["iteration 0: first draft", "iteration 1: revision 1"]
    assert len(fig.layout.shapes) == 2  # one shaded band per review iteration
    bars = {t.name: t for t in fig.data if t.type == "bar"}
    first = bars["steps · first draft"]
    research_bars = [(b, b + w) for y, b, w in zip(first.y, first.base, first.x, strict=True)
                     if y in research]
    assert len(research_bars) == 5
    assert max(s for s, _ in research_bars) < min(e for _, e in research_bars)  # overlap


# ------------------------------------------------------------------ AC6.2


def test_ac62_crewai_timeline_shows_roles_async_overlap_and_guardrail_retries(
        make: Make, slow_research: Callable[..., AgentSuite], repo: Repository,
        monkeypatch: pytest.MonkeyPatch) -> None:
    agents = slow_research(0.4, composer=HeatDefectComposer())
    result = make(PlannerMode.CREWAI_SEQUENTIAL, agents=agents).run(SUMMER, "s-62",
                                                                     today=TODAY)
    assert result.status is PlanStatus.APPROVED and result.iterations == 1
    trace = stored(repo, result.run_id)
    rows = ob.timeline_rows(trace)
    lanes = {r.lane for r in rows}
    assert {"Attractions Specialist", "Dining Specialist", "Itinerary Architect",
            "Quality Reviewer", "Budget Controller"} <= lanes  # tasks per role
    research = set(ob.RESEARCH_STEPS)
    overlapping, pairs = lane_overlap(rows, research, "crew_task")
    assert pairs == 10 and overlapping == 10  # async research tasks overlap
    compose_task = next(r for r in rows if r.name == "compose_itinerary"
                        and r.kind == "crew_task")
    assert compose_task.lane == "Itinerary Architect" and "1 guardrail retry" in \
        compose_task.detail
    stages = [r for r in rows if r.name == "compose_itinerary" and r.kind == "stage"]
    assert [r.iteration for r in stages] == [0, 1]  # each attempt is its own bar
    rejected = [m for m in ob.markers(trace) if m.kind == "guardrail_rejected"]
    assert len(rejected) == 1 and rejected[0].lane == "Itinerary Architect"
    assert "heat.midday_outdoor" in rejected[0].label
    assert ob.run_metrics(trace).guardrail_rejections == 1
    assert [b.iteration for b in ob.iteration_bands(rows)] == [0, 1]
    fig = figure_parts(trace, monkeypatch)
    names = [t.name for t in fig.data]
    assert "CrewAI task window" in names and "Guardrail rejected → retry" in names
    assert any(t.name == "LLM call" for t in fig.data)


def test_hierarchical_timeline_shows_delegations_and_one_by_one_research(
        make: Make, repo: Repository) -> None:
    result = make(PlannerMode.CREWAI_HIERARCHICAL).run(WINTER, "s-hier", today=TODAY)
    trace = stored(repo, result.run_id)
    marks = [m for m in ob.markers(trace) if m.kind == "delegation"]
    assert len(marks) >= 9 and {m.lane for m in marks} == {"Trip Manager"}
    # Every event lies inside the run's own window (CrewAI tool events are naive local
    # time; read as UTC they landed hours away on any machine east/west of UTC).
    run = trace.run
    assert run.ended_at is not None
    for e in trace.events:
        for t in (e.started_at, e.ended_at or e.started_at):
            low, high = run.started_at - timedelta(seconds=2), run.ended_at + timedelta(seconds=2)
            assert low <= t <= high, (e.agent, e.event, t)
    assert max(m.at_s for m in marks) <= (run.duration_s or 0) + 2
    par = ob.parallelism(ob.step_spans(trace))
    assert par is not None and not par.parallel  # by design: one manager executor
    lanes = {r.lane for r in ob.timeline_rows(trace) if r.kind == "stage"}
    assert "Attractions Specialist" in lanes  # the specialist did the work


# ------------------------------------------------------------------ AC6.3


class BlockingComposer:
    """Real composer that waits for the test before composing (a live run to watch)."""

    def __init__(self) -> None:
        self.entered, self.release = threading.Event(), threading.Event()

    def __call__(self, data: composer.ComposerInput, llm: Any, tools: Toolbox) -> Itinerary:
        self.entered.set()
        assert self.release.wait(60), "test never released the composer"
        return composer.run(data, llm, tools)


def wait_for(predicate: Callable[[], bool], timeout: float = 30) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached")


@pytest.mark.parametrize("mode", [PlannerMode.LANGGRAPH, PlannerMode.CREWAI_SEQUENTIAL],
                         ids=lambda m: m.value)
def test_ac63_live_status_from_sqlite_highlights_the_running_step(
        mode: PlannerMode, make: Make, repo: Repository) -> None:
    blocker = BlockingComposer()
    planner = make(mode, agents=AgentSuite(composer=blocker))
    outcome: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: outcome.setdefault(
        "result", planner.run(WINTER, f"s-live-{mode.value}", today=TODAY)), daemon=True)
    worker.start()
    try:
        assert blocker.entered.wait(60)
        [run] = [r for r in repo.list_runs(f"s-live-{mode.value}")]
        spec = (ob.langgraph_spec(describe_graph(planner.graph))
                if mode is PlannerMode.LANGGRAPH else ob.crewai_spec(planner.factory.describe()))

        def status() -> dict[str, ob.StepStatus]:
            trace = ob.load_trace(repo, run.id)  # SQLite only, like the page
            assert trace is not None and trace.live
            return ob.step_status(ob.step_spans(trace), spec.steps)

        wait_for(lambda: status()["compose_itinerary"] is ob.StepStatus.RUNNING)
        now = status()
        assert now["parse_request"] is ob.StepStatus.DONE
        assert all(now[s] is ob.StepStatus.DONE for s in ob.RESEARCH_STEPS)
        assert now["allocate_budget"] is ob.StepStatus.DONE
        review = "review" if mode is PlannerMode.LANGGRAPH else "review_itinerary"
        assert now[review] is ob.StepStatus.PENDING
        dot = ob.to_dot(spec, now)
        assert '"compose_itinerary" [label="compose_itinerary' in dot and "◐ running" in dot
        assert ob.current_steps(ob.step_spans(ob.load_trace(repo, run.id))) \
            == ["compose_itinerary"]
    finally:
        blocker.release.set()
        worker.join(60)
    assert outcome["result"].status is PlanStatus.APPROVED
    final = ob.load_trace(repo, run.id)
    assert final is not None and not final.live
    done = ob.step_status(ob.step_spans(final), spec.steps)
    assert done["compose_itinerary"] is ob.StepStatus.DONE and done[review] \
        is ob.StepStatus.DONE


def test_ac63_every_step_transition_is_written_immediately(
        make: Make, repo: Repository, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each node start/end closes a write batch, so a 1 s poller sees every transition."""
    batches: list[list[Any]] = []
    original = repo.add_trace_events

    def spy(events: list[Any]) -> None:
        batches.append(list(events))
        original(events)

    monkeypatch.setattr(repo, "add_trace_events", spy)
    result = make(PlannerMode.LANGGRAPH).run(WINTER, "s-batches", today=TODAY)
    trace = stored(repo, result.run_id)
    transitions = [e for e in trace.events if e.payload.get("kind") == "node"
                   and e.event in (TraceEventType.START, TraceEventType.END)]
    last_of_batch = {(b[-1].agent, b[-1].event, b[-1].started_at) for b in batches}
    assert len(transitions) >= 2 * 12
    assert all((e.agent, e.event, e.started_at) in last_of_batch for e in transitions)
    assert sum(len(b) for b in batches) == len(trace.events)


# ------------------------------------------------------------------ AC6.4


@pytest.mark.parametrize("mode", list(PlannerMode), ids=lambda m: m.value)
def test_ac64_every_step_output_validates_as_its_model(mode: PlannerMode, make: Make,
                                                        repo: Repository) -> None:
    result = make(mode).run(WINTER, f"s-64-{mode.value}", today=TODAY)
    trace = stored(repo, result.run_id)
    with_output = [e for e in trace.events if "output_schema" in e.payload]
    schemas = {e.payload["output_schema"] for e in with_output}
    assert {"RequirementsResult", "SelectionResult[AttractionOption]", "BudgetPlan",
            "Itinerary", "ReviewReport", "TransportPlan", "WeatherEventsResult"} <= schemas
    for e in with_output:
        checked = ob.validate_output(e.payload)
        assert checked is not None and checked.valid is True, (e.agent, checked)
    compose = next(e for e in with_output if e.payload["output_schema"] == "Itinerary")
    view = ob.inspect_event(compose, trace.origin)
    assert view.output is not None and view.output.valid
    rebuilt = Itinerary.model_validate(view.output.data)
    assert rebuilt == result.itinerary or result.iterations > 0
    assert view.inputs and "request" in view.inputs and view.header["duration (s)"] >= 0
    calls = [e for e in trace.events if e.event is TraceEventType.LLM_CALL
             and e.payload.get("prompt")]
    assert calls, "LLM calls should carry a prompt preview"
    llm_view = ob.inspect_event(calls[0], trace.origin)
    assert llm_view.prompt[0]["role"] == "system" and llm_view.header["tokens in"] > 0


# ------------------------------------------------------------------ AC6.5


def test_ac65_history_is_rebuilt_from_sqlite_in_a_new_process(
        make: Make, repo: Repository, tmp_path: Path) -> None:
    result = make(PlannerMode.CREWAI_SEQUENTIAL).run(WINTER, "s-65", today=TODAY)
    here = ob.run_metrics(stored(repo, result.run_id))
    script = (
        "import json, sys\n"
        "from dubai_planner.storage import Repository\n"
        "from dubai_planner.services import observability as ob\n"
        "repo = Repository(sys.argv[1])\n"
        "t = ob.load_trace(repo, sys.argv[2])\n"
        "m = ob.run_metrics(t)\n"
        "rows = ob.timeline_rows(t)\n"
        "print(json.dumps({'events': len(t.events), 'llm_calls': m.llm_calls,"
        " 'tokens': m.tokens, 'iterations': m.review_iterations, 'outcome': m.outcome,"
        " 'rows': len(rows), 'parallel': m.research.parallel,"
        " 'attempts': len(ob.recorded_attempts(t))}))\n")
    out = subprocess.run([sys.executable, "-c", script, str(repo.db_path), result.run_id],  # noqa: S603
                         capture_output=True, text=True, check=True, timeout=120)
    fresh = json.loads(out.stdout.strip().splitlines()[-1])
    assert fresh == {"events": here.events, "llm_calls": here.llm_calls,
                     "tokens": here.tokens, "iterations": here.review_iterations,
                     "outcome": "approved", "rows": len(ob.timeline_rows(stored(
                         repo, result.run_id))), "parallel": True, "attempts": 1}
    run = repo.get_run(result.run_id)
    assert run is not None and run.tokens == run.tokens_in + run.tokens_out > 0
    assert run.iterations == 0 and run.llm_latency_s >= 0
    assert run.context["provider"] == "mock" and run.context["process"] == "sequential"


# ------------------------------------------------------------------ failures


def test_failed_step_is_shown_failed_and_the_run_is_reported(make: Make,
                                                             repo: Repository) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("the budget table is unreadable")

    planner: LangGraphPlanner = make(PlannerMode.LANGGRAPH, agents=AgentSuite(budget=broken))
    result = planner.run(WINTER, "s-fail", today=TODAY)
    assert result.status is PlanStatus.FAILED
    trace = stored(repo, result.run_id)
    spec = ob.langgraph_spec(describe_graph(planner.graph))
    status = ob.step_status(ob.step_spans(trace), spec.steps)
    assert status["allocate_budget"] is ob.StepStatus.FAILED
    assert status["compose_itinerary"] is ob.StepStatus.PENDING
    assert status["research_dining"] is ob.StepStatus.DONE
    m = ob.run_metrics(trace)
    assert m.status == "failed" and m.outcome == "failed" and m.total_aed is None
    assert "✖ failed" in ob.to_dot(spec, status)


def test_cancelled_run_leaves_stopped_steps_not_running(make: Make,
                                                        repo: Repository) -> None:
    planner: CrewAIPlanner = make(PlannerMode.CREWAI_SEQUENTIAL)
    stream = planner.stream(WINTER, "s-cancel", today=TODAY)
    first = next(stream)
    stream.close()  # the user left: the crew stops at its next tool/guardrail
    trace = stored(repo, first.run_id)
    assert trace.run.status is RunStatus.INTERRUPTED and not trace.live
    spans = ob.step_spans(trace)
    assert all(s.status is not ob.StepStatus.RUNNING for s in spans)
    assert ob.run_metrics(trace).duration_s is not None


def test_run_without_trace_events_is_reported_as_missing(repo: Repository) -> None:
    run = repo.start_run(PlannerMode.LANGGRAPH)
    repo.finish_run(run.id, RunStatus.SUCCEEDED, llm_calls=2, tokens=50)
    trace = ob.load_trace(repo, run.id)
    assert trace is not None and trace.events == [] and ob.timeline_rows(trace) == []
    m = ob.run_metrics(trace)
    assert m.llm_calls == 2 and m.research is None and m.steps is None
    assert ob.load_trace(repo, "does-not-exist") is None


def test_running_record_of_a_dead_process_is_not_live(make: Make, repo: Repository) -> None:
    run = repo.start_run(PlannerMode.LANGGRAPH)  # never finished: the process was killed
    trace = ob.load_trace(repo, run.id, active=False)
    assert trace is not None and not trace.live


# ------------------------------------------------------------------ comparison


def test_same_request_in_both_frameworks_is_compared_like_for_like(
        make: Make, repo: Repository) -> None:
    lg = make(PlannerMode.LANGGRAPH).run(WINTER, "s-cmp-lg", today=TODAY)
    cr = make(PlannerMode.CREWAI_SEQUENTIAL).run(WINTER, "s-cmp-cr", today=TODAY)
    a, b = stored(repo, lg.run_id), stored(repo, cr.run_id)
    checks = ob.fairness(a, b)
    assert all(c.same is True for c in checks), [(c.label, c.a, c.b) for c in checks]
    rows = {r.metric: r for r in ob.compare(ob.run_metrics(a), ob.run_metrics(b))}
    ma, mb = ob.run_metrics(a), ob.run_metrics(b)
    assert rows["LLM calls"].delta == mb.llm_calls - ma.llm_calls
    assert rows["Total tokens"].a == lg_tokens(repo, lg.run_id)
    assert rows["Research execution"].a.startswith("parallel")
    assert rows["Research execution"].b.startswith("parallel")
    assert rows["Itinerary total (AED, estimate)"].delta == 0  # one shared agent core
    assert rows["Delegations"].a == 0 and rows["Tool calls"].b >= 9
    assert ob.messages_by_run(repo, repo.list_runs())[lg.run_id] == WINTER


def lg_tokens(repo: Repository, run_id: str) -> int:
    run = repo.get_run(run_id)
    assert run is not None
    return run.tokens


def test_different_requests_are_flagged_unfair(make: Make, repo: Repository) -> None:
    a = make(PlannerMode.LANGGRAPH).run(WINTER, "s-u1", today=TODAY)
    b = make(PlannerMode.CREWAI_SEQUENTIAL).run(SUMMER, "s-u2", today=TODAY)
    checks = {c.label: c.same for c in ob.fairness(stored(repo, a.run_id),
                                                   stored(repo, b.run_id))}
    assert checks["Same user message"] is False and checks["Same trip request"] is False
    assert checks["Same seed dataset (fingerprint)"] is True


# ------------------------------------------------------------------ secrets


def test_no_secret_reaches_the_stored_trace_or_its_export(
        make: Make, repo: Repository, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", CANARY)
    get_settings.cache_clear()

    def leaky(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"401 Unauthorized: invalid api key {CANARY}")

    result = make(PlannerMode.LANGGRAPH, agents=AgentSuite(budget=leaky),
                  settings=get_settings()).run(WINTER, "s-secret", today=TODAY)
    assert result.status is PlanStatus.FAILED
    import sqlite3

    with sqlite3.connect(repo.db_path) as conn:
        dump = "\n".join(conn.iterdump())
    assert CANARY not in dump and "***" in dump  # the error was recorded, redacted
    trace = stored(repo, result.run_id)
    for i, e in enumerate(trace.events):
        assert CANARY not in str(ob.inspect_event(e, trace.origin)), i
    assert all(CANARY not in json.dumps(e.payload) for e in trace.events)
