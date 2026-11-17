"""Phase 6 analysis on synthetic traces: spans, status, timeline, parallelism, metrics,
comparison, inspector, graph rendering, safe payloads and the trace recorder.

Integration tests with real (mock-LLM) runs of both frameworks are in
``tests/integration/test_observability_runs.py``.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from dubai_planner.config import get_settings
from dubai_planner.models import (
    PlannerMode,
    RunRecord,
    RunStatus,
    TraceEvent,
    TraceEventType,
)
from dubai_planner.orchestration.common import RunStats, data_fingerprint, redact_event
from dubai_planner.services import observability as ob
from dubai_planner.storage import Repository
from dubai_planner.tracing.payloads import (
    MAX_OUTPUT_CHARS,
    output_payload,
    record_io,
    schema_name,
    summarize,
)

T0 = datetime(2027, 2, 10, 9, 0, tzinfo=UTC)
CANARY = "gsk_CANARY_trace_never_stores_this_42"
E = TraceEventType


def at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def span(agent: str, start: float, end: float | None, *, kind: str = "node", it: int = 0,
         error: bool = False, run_id: str = "r1", mode: str = "langgraph",
         **payload: Any) -> list[TraceEvent]:
    """START (+ END/ERROR) events exactly as ``agent_span`` publishes them."""
    p = {"kind": kind, **payload}
    events = [TraceEvent(run_id=run_id, mode=mode, agent=agent, event=E.START,
                         started_at=at(start), iteration=it, payload={"kind": kind})]
    if end is not None:
        events.append(TraceEvent(run_id=run_id, mode=mode, agent=agent,
                                 event=E.ERROR if error else E.END, started_at=at(start),
                                 ended_at=at(end), iteration=it, payload=p))
    return events


def llm(agent: str, t: float, tin: int, tout: int, *, it: int = 0) -> TraceEvent:
    return TraceEvent(run_id="r1", mode="langgraph", agent=agent, event=E.LLM_CALL,
                      started_at=at(t), ended_at=at(t + 0.5), tokens_in=tin, tokens_out=tout,
                      iteration=it, payload={"schema": "X", "latency_s": 0.5})


def trace(events: list[TraceEvent], *, status: RunStatus = RunStatus.SUCCEEDED,
          mode: PlannerMode = PlannerMode.LANGGRAPH, active: bool | None = None,
          **run: Any) -> ob.RunTrace:
    record = RunRecord(id="r1", session_id="s1", mode=mode, status=status, started_at=T0,
                       **run)
    return ob.RunTrace(run=record, events=events, active=active)


RESEARCH = ob.RESEARCH_STEPS


def langgraph_events(*, parallel: bool = True, refine: bool = True) -> list[TraceEvent]:
    ev = [*span("parse_request", 0, 1), llm("requirements", 0.2, 100, 20)]
    for i, name in enumerate(RESEARCH):
        start = 1.1 if parallel else 1.1 + i
        ev += span(name, start, start + 0.9)
    t = 2.2 if parallel else 6.2
    ev += span("merge_research", t, t + 0.1) + span("allocate_budget", t + 0.1, t + 0.3)
    ev += [*span("compose_itinerary", t + 0.3, t + 1.0), llm("composer", t + 0.4, 500, 200)]
    ev += span("validate", t + 1.0, t + 1.1) + span("review", t + 1.1, t + 1.5)
    if refine:
        ev += span("refine_itinerary", t + 1.5, t + 1.6)
        ev += span("compose_itinerary", t + 1.6, t + 2.2, it=1)
        ev += [llm("composer", t + 1.7, 600, 210, it=1)]
        ev += span("validate", t + 2.2, t + 2.3, it=1) + span("review", t + 2.3, t + 2.6, it=1)
    ev += span("finalize", t + 2.7, t + 2.8, it=1 if refine else 0, status="approved")
    return ev


# ------------------------------------------------------------------ spans and status


def test_spans_pair_start_and_end_in_start_order() -> None:
    spans = ob.build_spans(langgraph_events(), kinds={"node"})
    assert [s.name for s in spans][:2] == ["parse_request", "research_attractions"]
    assert all(s.status is ob.StepStatus.DONE and s.end is not None for s in spans)
    assert {s.name for s in spans if s.iteration == 1} == {"compose_itinerary", "validate",
                                                          "review", "finalize"}
    assert spans[0].duration_s == pytest.approx(1.0)


def test_open_span_is_running_while_live_and_stopped_after() -> None:
    events = span("parse_request", 0, 1) + span("compose_itinerary", 1, None)
    live = ob.build_spans(events, live=True)
    assert live[-1].status is ob.StepStatus.RUNNING and live[-1].end is None
    assert ob.build_spans(events, live=False)[-1].status is ob.StepStatus.STOPPED
    assert ob.current_steps(live) == ["compose_itinerary"]


def test_interrupted_node_is_waiting_not_running() -> None:
    events = span("parse_request", 0, 1) + span("ask_clarification", 1, None)
    events.append(TraceEvent(run_id="r1", mode="langgraph", agent="ask_clarification",
                             event=E.INFO, started_at=at(1.1),
                             payload={"kind": "node", "interrupted": True}))
    [_, ask] = ob.build_spans(events, live=True)
    assert ask.status is ob.StepStatus.WAITING


def test_step_status_pending_running_done_failed_and_latest_iteration_wins() -> None:
    events = langgraph_events()[:-2]  # without finalize
    events += span("finalize", 9, 9.5, error=True)
    status = ob.step_status(ob.build_spans(events), ["parse_request", "review", "finalize",
                                                     "ask_clarification"])
    assert status == {"parse_request": ob.StepStatus.DONE, "review": ob.StepStatus.DONE,
                      "finalize": ob.StepStatus.FAILED,
                      "ask_clarification": ob.StepStatus.PENDING}
    # A step re-running in a later iteration shows its newest state.
    rerun = span("review", 0, 1) + span("review", 2, None, it=1)
    assert ob.step_status(ob.build_spans(rerun, live=True), ["review"])["review"] \
        is ob.StepStatus.RUNNING


def test_crewai_task_span_wins_over_its_stage_span_and_lanes_are_roles() -> None:
    ev = span("compose_itinerary", 0, 5, kind="crew_task", mode="crewai_sequential",
              role_key="itinerary_architect", executed_by="Itinerary Architect")
    ev += span("compose_itinerary", 1, 2, kind="stage", mode="crewai_sequential",
               role="Itinerary Architect")
    spans = ob.build_spans(ev, kinds=ob.STEP_KINDS)
    assert {s.lane for s in spans} == {"Itinerary Architect"}
    status = ob.step_status(spans, ["compose_itinerary"])
    assert status["compose_itinerary"] is ob.StepStatus.DONE


# ------------------------------------------------------------------ timeline


def test_timeline_rows_are_relative_and_ordered_with_one_band_per_iteration() -> None:
    tr = trace(langgraph_events())
    rows = ob.timeline_rows(tr)
    assert rows[0].start_s == 0 and rows[0].name == "parse_request"
    assert [r.start_s for r in rows] == sorted(r.start_s for r in rows)
    compose = [r for r in rows if r.name == "compose_itinerary"]
    assert [r.iteration for r in compose] == [0, 1]
    assert compose[0].tokens == 700 and compose[1].tokens == 810
    bands = ob.iteration_bands(rows)
    assert [(b.iteration, b.label) for b in bands] == [(0, "first draft"), (1, "revision 1")]
    assert bands[0].end_s <= bands[1].start_s


def test_running_bar_extends_to_now_and_markers_are_attributed() -> None:
    ev = span("parse_request", 0, 1) + span("compose_itinerary", 1, None)
    ev.append(llm("composer", 1.2, 10, 5))
    tr = trace(ev, status=RunStatus.RUNNING, active=True)
    rows = ob.timeline_rows(tr, now=at(4))
    assert rows[-1].status is ob.StepStatus.RUNNING and rows[-1].end_s == 4.0
    [m] = ob.markers(tr)
    assert m.lane == "compose_itinerary" and m.kind == "llm_call" and m.tokens == 15


def test_guardrail_rejections_and_delegations_become_markers() -> None:
    mode = "crewai_hierarchical"
    ev = span("compose_itinerary", 0, 3, kind="stage", mode=mode, role="Itinerary Architect")
    ev.append(TraceEvent(run_id="r1", mode=mode, agent="compose_itinerary",
                         event=E.GUARDRAIL, started_at=at(1), payload={
                             "kind": "guardrail", "passed": False,
                             "rules": ["heat.midday_outdoor"]}))
    ev.append(TraceEvent(run_id="r1", mode=mode, agent="trip_manager", event=E.DELEGATION,
                         started_at=at(0.5), payload={"kind": "delegation",
                                                      "role": "Trip Manager",
                                                      "coworker": "Itinerary Architect"}))
    marks = {m.kind: m for m in ob.markers(trace(ev, mode=PlannerMode.CREWAI_HIERARCHICAL))}
    assert marks["guardrail_rejected"].lane == "Itinerary Architect"
    assert "heat.midday_outdoor" in marks["guardrail_rejected"].label
    assert marks["delegation"].lane == "Trip Manager"


# ------------------------------------------------------------------ parallelism


def test_parallelism_is_measured_from_timestamps() -> None:
    par = ob.parallelism(ob.build_spans(langgraph_events(parallel=True)))
    assert par is not None and par.parallel and par.overlapping_pairs == par.pairs == 10
    assert par.window_s == pytest.approx(0.9) and par.sum_s == pytest.approx(4.5)
    assert par.speedup == pytest.approx(5.0) and par.label.startswith("parallel")
    seq = ob.parallelism(ob.build_spans(langgraph_events(parallel=False)))
    assert seq is not None and not seq.parallel and seq.label == "sequential (no overlap)"
    assert seq.speedup == pytest.approx(0.9 * 5 / 4.9, abs=0.01)


def test_parallelism_partial_and_missing() -> None:
    ev = span("research_attractions", 0, 2) + span("research_dining", 1, 3)
    ev += span("research_transport", 4, 5)
    par = ob.parallelism(ob.build_spans(ev))
    assert par is not None and par.overlapping_pairs == 1 and par.pairs == 3
    assert par.label.startswith("partly parallel")
    assert ob.parallelism(ob.build_spans(span("parse_request", 0, 1))) is None
    # Failed or unfinished research steps are not counted as measured work.
    assert ob.parallelism(ob.build_spans(span("research_dining", 0, None))) is None


# ------------------------------------------------------------------ metrics / comparison


def test_run_metrics_from_record_and_events() -> None:
    m = ob.run_metrics(trace(langgraph_events(), iterations=1, duration_s=5.2, llm_calls=3,
                             tokens=1630, tokens_in=1200, tokens_out=430, llm_latency_s=1.5))
    assert (m.duration_s, m.llm_calls, m.tokens, m.review_iterations) == (5.2, 3, 1630, 1)
    assert m.guardrail_rejections == 0 and m.research is not None and m.research.parallel
    assert m.outcome == "approved" and m.steps == 16


def test_run_metrics_without_events_invent_nothing() -> None:
    m = ob.run_metrics(trace([], duration_s=3.0, llm_calls=4, tokens=900))
    assert (m.duration_s, m.llm_calls, m.tokens) == (3.0, 4, 900)
    assert m.research is None and m.guardrail_rejections is None and m.steps is None
    assert m.review_iterations is None and m.llm_latency_s is None and m.events == 0


def test_run_metrics_of_incomplete_trace_derive_only_what_exists() -> None:
    ev = [*span("parse_request", 0, 1), llm("requirements", 0.2, 50, 10)]
    m = ob.run_metrics(trace(ev, status=RunStatus.INTERRUPTED))
    assert m.llm_calls == 1 and m.tokens == 60 and m.llm_latency_s == 0.5
    assert m.duration_s == pytest.approx(1.0) and m.research is None
    assert m.outcome == "needs clarification / interrupted"


def test_compare_deltas_only_when_both_sides_were_measured() -> None:
    a = ob.run_metrics(trace(langgraph_events(), duration_s=5.0, llm_calls=8, tokens=100))
    b = ob.run_metrics(trace([], mode=PlannerMode.CREWAI_SEQUENTIAL, duration_s=7.5,
                             llm_calls=17, tokens=300))
    rows = {r.metric: r for r in ob.compare(a, b)}
    assert rows["Wall-clock time (s)"].delta == 2.5
    assert rows["LLM calls"].delta == 9 and rows["Total tokens"].delta == 200
    assert rows["Guardrail rejections"].b is None and rows["Guardrail rejections"].delta is None
    assert rows["Research execution"].a.startswith("parallel")
    assert rows["Research execution"].b is None
    assert rows["Outcome"].delta is None


def _with_request(events: list[TraceEvent], budget: str) -> list[TraceEvent]:
    out = []
    for e in events:
        if e.agent == "parse_request" and e.event is E.END:
            e = e.model_copy(update={"payload": {**e.payload, "output": {"request": {
                "start_date": "2027-02-10", "end_date": "2027-02-12",
                "travellers": {"adults": 2}, "budget_aed": budget}}}})
        out.append(e)
    return out


def test_fairness_checks_same_different_and_unknown() -> None:
    ctx = {"provider": "mock", "model": "mock", "data": "abc", "max_review_iterations": 2,
           "http_offline": True}
    a = ob.RunTrace(run=RunRecord(id="a", session_id="s", mode=PlannerMode.LANGGRAPH,
                                  context=ctx), events=_with_request(langgraph_events(), "6000"),
                    message="2 adults")
    b = ob.RunTrace(run=RunRecord(id="b", session_id="t", mode=PlannerMode.CREWAI_SEQUENTIAL,
                                  context=ctx), events=_with_request(langgraph_events(), "6000"),
                    message="2 adults")
    assert all(c.same for c in ob.fairness(a, b))
    c = ob.RunTrace(run=RunRecord(id="c", session_id="u", mode=PlannerMode.LANGGRAPH,
                                  context=ctx | {"data": "other"}),
                    events=_with_request(langgraph_events(), "9000"), message="2 adults")
    checks = {x.label: x.same for x in ob.fairness(a, c)}
    assert checks["Same trip request"] is False and checks["Same seed dataset (fingerprint)"] \
        is False and checks["Same user message"] is True
    old = ob.RunTrace(run=RunRecord(id="d", session_id=None, mode=PlannerMode.LANGGRAPH),
                      events=[])
    assert all(x.same is None for x in ob.fairness(a, old))


# ------------------------------------------------------------------ inspector


def test_validate_output_valid_tampered_unknown_and_truncated() -> None:
    from factories import itinerary as make_itinerary

    payload = output_payload(make_itinerary())
    assert payload["output_schema"] == "Itinerary"
    ok = ob.validate_output(payload)
    assert ok is not None and ok.valid is True and ok.data["days"]
    bad = dict(payload, output={**payload["output"], "days": "not a list"})
    checked = ob.validate_output(bad)
    assert checked is not None and checked.valid is False and checked.error
    unknown = ob.validate_output({"output_schema": "Mystery", "output": {"x": 1}})
    assert unknown is not None and unknown.valid is None
    big = ob.validate_output({"output_schema": "Itinerary", "output_truncated": True,
                              "output_summary": {"days": "3 item(s)"}})
    assert big is not None and big.valid is None and "too large" in str(big.error)
    assert ob.validate_output({"kind": "node"}) is None
    empty = ob.validate_output(output_payload([]))
    assert empty is not None and empty.valid is True


def test_inspect_event_splits_payload_and_redacts_secrets(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", CANARY)
    get_settings.cache_clear()
    event = TraceEvent(run_id="r1", mode="langgraph", agent="composer", event=E.LLM_CALL,
                       started_at=at(1), ended_at=at(2), tokens_in=5, tokens_out=6, payload={
                           "latency_s": 1.0, "prompt": [{"role": "user", "chars": 3,
                                                         "preview": f"key {CANARY}"}],
                           "response_preview": "{}", "error": f"bad key {CANARY}"})
    view = ob.inspect_event(event, T0)
    assert view.header["tokens in"] == 5 and view.header["LLM latency (s)"] == 1.0
    assert view.header["duration (s)"] == 1.0 and view.output is None
    assert CANARY not in str(view) and "***" in view.prompt[0]["preview"]
    assert "prompt" not in view.details and "error" in view.details


def test_state_changes_and_recorded_attempts() -> None:
    ev = [TraceEvent(run_id="r1", mode="langgraph", agent="review", event=E.INFO,
                     started_at=at(1), payload={"kind": "state_update", "update": {"a": 1}})]
    ev += span("review", 1, 2, attempt={"approved": False, "errors": 1},
               output={"approved": False, "issues": [{"severity": "error", "rule_id":
                                                      "budget.total", "day_index": None,
                                                      "message": "over budget"}]})
    ev.append(TraceEvent(run_id="r1", mode="crewai_sequential", agent="compose_itinerary",
                         event=E.GUARDRAIL, started_at=at(3), iteration=0, payload={
                             "kind": "guardrail", "passed": False,
                             "attempt": {"approved": False}, "issues": [{"message": "hot"}]}))
    tr = trace(ev)
    assert ob.state_changes(tr)[0]["step"] == "review"
    attempts = ob.recorded_attempts(tr)
    assert [a["issues"][0]["message"] for a in attempts] == ["over budget", "hot"]


# ------------------------------------------------------------------ graph rendering


def test_dot_and_mermaid_show_status_in_words_and_highlight_running() -> None:
    spec = ob.langgraph_spec({"nodes": ["__start__", "parse_request", *RESEARCH,
                                        "merge_research", "__end__"],
                              "edges": [{"source": "__start__", "target": "parse_request",
                                         "conditional": False},
                                        {"source": "parse_request",
                                         "target": "research_dining", "conditional": True}],
                              "parallel": list(RESEARCH)})
    status = {"parse_request": ob.StepStatus.DONE, "research_dining": ob.StepStatus.RUNNING}
    dot = ob.to_dot(spec, status)
    assert "cluster_parallel" in dot and "parallel research" in dot
    assert '"research_dining" [label="research_dining\\n◐ running"' in dot
    assert "penwidth=3" in dot and "● done" in dot and "○ pending" in dot
    assert '"parse_request" -> "research_dining" [style=dashed]' in dot
    mermaid = ob.to_mermaid(spec, status)
    assert "class research_dining running" in mermaid and "-.->" in mermaid
    assert spec.steps == ["parse_request", *RESEARCH, "merge_research"]


def test_crewai_spec_draws_roles_context_async_and_manager() -> None:
    desc = {"process": "hierarchical", "manager": "Trip Manager", "tasks": [
        {"name": "parse_request", "role": "Requirements Analyst", "async": False,
         "context": []},
        {"name": "research_dining", "role": "Dining Specialist", "async": False,
         "context": []},
        {"name": "allocate_budget", "role": "Budget Controller", "async": False,
         "context": ["research_dining"]},
        {"name": "compose_itinerary", "role": "Itinerary Architect", "async": False,
         "context": ["allocate_budget", "research_dining"]},
        {"name": "review_itinerary", "role": "Quality Reviewer", "async": False,
         "context": ["compose_itinerary"]}]}
    spec = ob.crewai_spec(desc)
    dot = ob.to_dot(spec)
    assert "Dining Specialist" in dot and "manager\\nTrip Manager" in dot
    assert '"compose_itinerary" -> "compose_itinerary" [style=dashed, label="guardrail retry"]' \
        in dot
    assert "run one by one by the manager" in dot
    assert ("research_dining", "compose_itinerary", False, "context") not in spec.edges
    assert spec.title == "CrewAI crew (hierarchical)"


# ------------------------------------------------------------------ safe payloads


def test_payload_summaries_are_small_and_outputs_capped() -> None:
    from factories import itinerary as make_itinerary

    it = make_itinerary()
    s = summarize(it)
    assert isinstance(s, dict) and s["days"].endswith("item(s)")
    assert summarize("x" * 500).endswith("…") and len(summarize("x" * 500)) == 201
    assert schema_name([it]) == "list[Itinerary]" and schema_name([]) == "list[Any]"
    huge = {"blob": "y" * (MAX_OUTPUT_CHARS + 10)}
    out = output_payload(huge)
    assert out["output_truncated"] and "output" not in out and "output_summary" in out
    span_dict: dict[str, Any] = {}
    record_io(span_dict, inputs={"request": it}, output=it)
    assert span_dict["output_schema"] == "Itinerary" and "inputs" in span_dict


# ------------------------------------------------------------------ recorder


def _event(agent: str, event: TraceEventType, kind: str | None, **payload: Any) -> TraceEvent:
    p = {"kind": kind, **payload} if kind else dict(payload)
    return TraceEvent(run_id="r1", mode="langgraph", agent=agent, event=event,
                      started_at=at(0), payload=p)


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    repo = Repository(tmp_path / "app.db")
    repo.start_run(PlannerMode.LANGGRAPH, run_id="r1", context={"provider": "mock"})
    return repo


def test_recorder_writes_at_every_transition_and_on_flush(repo: Repository) -> None:
    stats = RunStats("r1", repo=repo)
    stats(_event("parse_request", E.START, "node"))
    assert repo.count_trace_events("r1") == 1  # a transition is written at once
    stats(_event("requirements", E.START, None))
    stats(TraceEvent(run_id="r1", mode="langgraph", agent="requirements", event=E.LLM_CALL,
                     tokens_in=7, tokens_out=3, payload={"latency_s": 0.25}))
    assert repo.count_trace_events("r1") == 1  # inner events wait for the next transition
    stats(_event("parse_request", E.END, "node"))
    assert repo.count_trace_events("r1") == 4
    stats(_event("x", E.GUARDRAIL, "guardrail", passed=False))
    assert repo.count_trace_events("r1") == 5
    stats(_event("x", E.INFO, None))
    assert stats.flush() == 1 and stats.flush() == 0
    stats(TraceEvent(run_id="other", mode="langgraph", agent="y", event=E.START))
    assert repo.count_trace_events("r1") == 6 and len(stats.events) == 6
    assert [e.agent for e in repo.list_trace_events("r1")][:2] == ["parse_request",
                                                                   "requirements"]
    assert (stats.llm_calls, stats.tokens_in, stats.tokens_out, stats.tokens) == (1, 7, 3, 10)
    assert stats.llm_latency_s == 0.25


def test_recorder_redacts_secrets_before_writing(repo: Repository,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", CANARY)
    get_settings.cache_clear()
    stats = RunStats("r1", repo=repo, settings=get_settings())
    stats(_event("composer", E.ERROR, "node", message=f"401 for key {CANARY}"))
    with sqlite3.connect(repo.db_path) as conn:
        raw = " ".join(r[0] for r in conn.execute("SELECT payload_json FROM trace_events"))
    assert CANARY not in raw and "***" in raw
    assert CANARY in stats.events[0].payload["message"]  # in memory only, never stored
    clean = redact_event(stats.events[0], get_settings())
    assert CANARY not in clean.model_dump_json()


def test_recorder_survives_a_failing_database(repo: Repository,
                                              monkeypatch: pytest.MonkeyPatch,
                                              caplog: pytest.LogCaptureFixture) -> None:
    def broken(_events: Any) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(repo, "add_trace_events", broken)
    stats = RunStats("r1", repo=repo)
    stats(_event("parse_request", E.START, "node"))  # must not raise
    assert stats.flush() == 0 and len(stats.events) == 1
    assert "could not persist" in caplog.text


def test_finish_kwargs_add_a_resumed_runs_prior_totals(repo: Repository) -> None:
    prior = repo.finish_run("r1", RunStatus.FAILED, llm_calls=2, tokens=30, tokens_in=20,
                            tokens_out=10, llm_latency_s=1.5, iterations=0)
    stats = RunStats("r1")
    stats(TraceEvent(run_id="r1", mode="langgraph", agent="a", event=E.LLM_CALL,
                     tokens_in=4, tokens_out=1, payload={"latency_s": 0.5}))
    assert stats.finish_kwargs(prior) == {"llm_calls": 3, "tokens": 35, "tokens_in": 24,
                                          "tokens_out": 11, "llm_latency_s": 2.0}
    assert stats.finish_kwargs(None)["llm_calls"] == 1


def test_run_record_keeps_aggregates_and_context(repo: Repository) -> None:
    repo.finish_run("r1", RunStatus.SUCCEEDED, llm_calls=3, tokens=12, tokens_in=8,
                    tokens_out=4, llm_latency_s=0.123456, iterations=2)
    run = repo.get_run("r1")
    assert run is not None and (run.tokens_in, run.tokens_out, run.iterations) == (8, 4, 2)
    assert run.llm_latency_s == 0.123 and run.context == {"provider": "mock"}


def test_data_fingerprint_changes_with_the_seed(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    first = data_fingerprint(str(tmp_path))
    data_fingerprint.cache_clear()
    (tmp_path / "a.json").write_text('{"x": 1}', encoding="utf-8")
    assert data_fingerprint(str(tmp_path)) != first and len(first) == 12
    data_fingerprint.cache_clear()


def test_crewai_naive_local_timestamps_become_utc() -> None:
    """CrewAI tool events carry naive local time; the listener must not read it as UTC."""
    from dubai_planner.orchestration.crewai.listener import _aware

    local_now = datetime.now()  # naive, local, like CrewAI's tool usage events
    assert abs((_aware(local_now) - datetime.now(UTC)).total_seconds()) < 5
    aware = datetime(2027, 2, 10, 9, 0, tzinfo=UTC)
    assert _aware(aware) == aware


def test_research_tile_label_is_short() -> None:
    assert ob.Parallelism(5, 1.0, 1.0, 0, 10).short == "sequential"
    assert ob.Parallelism(5, 1.0, 4.0, 10, 10).short == "parallel ×4.0"
