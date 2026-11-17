"""Execution analysis for the Phase 6 visualiser: spans, status, timeline, comparison.

Everything here is computed from what was *recorded*: the ``runs`` row and the run's
``trace_events`` in SQLite (written by ``orchestration.common.RunStats`` while the run
executes). Nothing is estimated or filled in: a value that the trace does not contain
is ``None`` and the UI shows it as missing. The module is framework-free (no LangGraph
or CrewAI import), so the same code reads both modes' traces:

- :func:`build_spans` pairs ``start``/``end``/``error`` events into :class:`Span`s;
  a ``start`` without an end is *running* while the run is, else *stopped*;
- :func:`step_status` colours graph nodes (pending / running / done / failed …);
- :func:`timeline_rows`, :func:`iteration_bands` and :func:`markers` feed the Gantt
  chart; :func:`parallelism` measures the research fan-out from timestamps;
- :func:`run_metrics`, :func:`compare` and :func:`fairness` drive run comparison;
- :func:`inspect_event` gives the inspector a safe view of one event, re-validating
  the recorded output against its Pydantic model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from functools import lru_cache
from itertools import combinations
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from dubai_planner.models import (
    ItineraryRecord,
    PlannerMode,
    RunRecord,
    RunStatus,
    TraceEvent,
    TraceEventType,
)
from dubai_planner.redaction import redact
from dubai_planner.storage import Repository

#: Span kinds that are one orchestration step (graph node, CrewAI task or its tool run).
STEP_KINDS = frozenset({"node", "crew_task", "stage"})
RESEARCH_STEPS = ("research_attractions", "research_accommodation", "research_dining",
                  "research_transport", "research_weather_events")
#: Steps of the review loop (both modes); one band per review iteration on the timeline.
LOOP_STEPS = frozenset({"compose_itinerary", "validate", "review", "review_itinerary",
                        "refine_itinerary"})
NO_TRACE = "No trace events were recorded for this run."


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    WAITING = "waiting"  # paused at a clarification question
    STOPPED = "stopped"  # started, never ended, and the run is over (cancelled/killed)


STATUS_ICON = {StepStatus.PENDING: "○", StepStatus.RUNNING: "◐", StepStatus.DONE: "●",
               StepStatus.FAILED: "✖", StepStatus.WAITING: "❓", StepStatus.STOPPED: "◌"}


# --------------------------------------------------------------------------- loading


@dataclass(frozen=True)
class RunTrace:
    """One run as stored: its record, events (publish order) and saved itinerary."""

    run: RunRecord
    events: list[TraceEvent]
    itinerary: ItineraryRecord | None = None
    message: str | None = None  # the user message that started the run
    #: Is the run executing now? ``None`` = trust the run record. A record left
    #: ``running`` by a killed process is not live: callers that know pass ``False``.
    active: bool | None = None

    @property
    def mode(self) -> PlannerMode:
        return self.run.mode

    @property
    def live(self) -> bool:
        if self.active is not None:
            return self.active and self.run.status is RunStatus.RUNNING
        return self.run.status is RunStatus.RUNNING

    @property
    def origin(self) -> datetime:
        """Time zero of the timeline: the first event, else the run's start."""
        starts = [e.started_at for e in self.events]
        return min(starts) if starts else self.run.started_at


def load_trace(repo: Repository, run_id: str, *, active: bool | None = None
               ) -> RunTrace | None:
    """The run's record and events from SQLite only (works after a restart)."""
    run = repo.get_run(run_id)
    if run is None:
        return None
    itinerary = message = None
    if run.session_id:
        itinerary = next((r for r in repo.list_itineraries(run.session_id)
                          if r.run_id == run_id), None)
        message = next((m.content for m in repo.list_messages(run.session_id)
                        if m.role == "user" and m.metadata.get("run_id") == run_id), None)
    return RunTrace(run=run, events=repo.list_trace_events(run_id), itinerary=itinerary,
                    message=message, active=active)


def list_run_traces(repo: Repository, session_id: str | None = None) -> list[RunRecord]:
    """Runs newest first (one session, or all when ``session_id`` is None)."""
    return sorted(repo.list_runs(session_id), key=lambda r: r.started_at, reverse=True)


def export_trace(trace: RunTrace) -> str:
    """The run's record and events as JSON, with secret values redacted."""
    data = {"run": trace.run.model_dump(mode="json"),
            "events": [json.loads(e.model_dump_json()) for e in trace.events]}
    return redact(json.dumps(data, indent=1, default=str))


def messages_by_run(repo: Repository, runs: Iterable[RunRecord]) -> dict[str, str]:
    """The user message that started each run (read once per session)."""
    out: dict[str, str] = {}
    for sid in {r.session_id for r in runs if r.session_id}:
        for m in repo.list_messages(sid):
            run_id = m.metadata.get("run_id")
            if m.role == "user" and run_id and run_id not in out:
                out[str(run_id)] = m.content
    return out


# --------------------------------------------------------------------------- spans


@dataclass(frozen=True)
class Span:
    name: str
    lane: str
    kind: str
    iteration: int
    start: datetime
    end: datetime | None
    status: StepStatus
    payload: dict[str, Any] = field(default_factory=dict)
    index: int = 0  # position of the closing (or opening) event in the run's events

    @property
    def duration_s(self) -> float | None:
        return None if self.end is None else (self.end - self.start).total_seconds()


def _lane(agent: str, kind: str, payload: dict[str, Any]) -> str:
    """Timeline lane: graph node (LangGraph); for CrewAI the role that did the work
    (the specialist running the stage tool, the agent executing the task)."""
    if kind in ("crew_task", "stage"):
        role = payload.get("role") or payload.get("executed_by") or payload.get("role_key")
        return str(role or agent)
    return agent


def build_spans(events: Sequence[TraceEvent], *, live: bool = False,
                kinds: Iterable[str] | None = None) -> list[Span]:
    """Pair start/end/error events into spans (in start order).

    A closing event carries its span's ``started_at``, so pairs are matched on
    (agent, kind, started_at). Unpaired starts are ``running`` while the run is
    ``live``, ``waiting`` when the step paused at an interrupt, else ``stopped``.
    """
    wanted = None if kinds is None else frozenset(kinds)
    opened: dict[tuple[str, str, datetime], tuple[int, TraceEvent]] = {}
    spans: list[Span] = []
    paused: set[str] = set()
    for i, e in enumerate(events):
        kind = str(e.payload.get("kind") or "agent")
        if e.event is TraceEventType.INFO and e.payload.get("interrupted"):
            paused.add(e.agent)
            continue
        if wanted is not None and kind not in wanted:
            continue
        key = (e.agent, kind, e.started_at)
        if e.event is TraceEventType.START:
            opened[key] = (i, e)
        elif e.event in (TraceEventType.END, TraceEventType.ERROR):
            first = opened.pop(key, None)
            start_event = first[1] if first else e
            payload = {**start_event.payload, **e.payload}
            status = StepStatus.FAILED if e.event is TraceEventType.ERROR else StepStatus.DONE
            spans.append(Span(
                name=e.agent, lane=_lane(e.agent, kind, payload), kind=kind,
                iteration=start_event.iteration, start=e.started_at,
                end=e.ended_at or e.started_at, status=status, payload=payload, index=i))
    for i, e in opened.values():
        kind = str(e.payload.get("kind") or "agent")
        status = (StepStatus.WAITING if e.agent in paused
                  else StepStatus.RUNNING if live else StepStatus.STOPPED)
        spans.append(Span(name=e.agent, lane=_lane(e.agent, kind, e.payload), kind=kind,
                          iteration=e.iteration,
                          start=e.started_at, end=None, status=status, payload=dict(e.payload),
                          index=i))
    return sorted(spans, key=lambda s: (s.start, s.index))


def step_spans(trace: RunTrace) -> list[Span]:
    """The orchestration steps of a run (nodes, or CrewAI tasks and stage tool runs)."""
    return build_spans(trace.events, live=trace.live, kinds=STEP_KINDS)


def step_status(spans: Sequence[Span], steps: Iterable[str]) -> dict[str, StepStatus]:
    """Each step's latest state (a step can run once per review iteration)."""
    out: dict[str, StepStatus] = {}
    for step in steps:
        own = [s for s in spans if s.name == step and s.kind in STEP_KINDS]
        if not own:
            out[step] = StepStatus.PENDING
            continue
        # Prefer the CrewAI task span over its tool's stage span when both exist.
        tasks = [s for s in own if s.kind != "stage"] or own
        out[step] = max(tasks, key=lambda s: (s.start, s.index)).status
    return out


def current_steps(spans: Sequence[Span]) -> list[str]:
    """Steps executing right now (a live run), in start order."""
    names: list[str] = []
    for s in spans:
        if s.status is StepStatus.RUNNING and s.kind in STEP_KINDS and s.name not in names:
            names.append(s.name)
    return names


# --------------------------------------------------------------------------- timeline


@dataclass(frozen=True)
class TimelineRow:
    lane: str
    name: str
    kind: str
    iteration: int
    start_s: float
    end_s: float
    status: StepStatus
    tokens: int = 0
    detail: str = ""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _rel(t: datetime, origin: datetime) -> float:
    return round((t - origin).total_seconds(), 3)


def timeline_rows(trace: RunTrace, *, now: datetime | None = None) -> list[TimelineRow]:
    """One bar per step span. A running span ends at ``now`` (or the last event)."""
    origin = trace.origin
    last = max((e.ended_at or e.started_at for e in trace.events), default=origin)
    tokens_by_iter = _llm_tokens(trace.events, trace.mode)
    rows: list[TimelineRow] = []
    for s in step_spans(trace):
        end = s.end or (now if now and trace.live else last)
        retries = s.payload.get("retries")
        detail = f"{s.kind}" + (f", {retries} guardrail retr{'y' if retries == 1 else 'ies'}"
                                if retries else "")
        rows.append(TimelineRow(
            lane=s.lane, name=s.name, kind=s.kind, iteration=s.iteration,
            start_s=_rel(s.start, origin), end_s=max(_rel(end, origin), _rel(s.start, origin)),
            status=s.status, tokens=tokens_by_iter.get((s.name, s.iteration), 0), detail=detail))
    return rows


def _llm_tokens(events: Sequence[TraceEvent], mode: PlannerMode) -> dict[tuple[str, int], int]:
    """Tokens of agent-core LLM calls, by (step name, iteration)."""
    out: dict[tuple[str, int], int] = {}
    for e in events:
        if e.event is TraceEventType.LLM_CALL:
            key = (_core_step(e.agent, mode), e.iteration)
            out[key] = out.get(key, 0) + e.tokens_in + e.tokens_out
    return out


@dataclass(frozen=True)
class Band:
    iteration: int
    start_s: float
    end_s: float

    @property
    def label(self) -> str:
        return "first draft" if self.iteration == 0 else f"revision {self.iteration}"


def iteration_bands(rows: Sequence[TimelineRow]) -> list[Band]:
    """One band per review iteration: the window of its compose → review steps."""
    by_iter: dict[int, list[TimelineRow]] = {}
    for r in rows:
        if r.name in LOOP_STEPS and r.kind != "crew_task":
            by_iter.setdefault(r.iteration, []).append(r)
    return [Band(it, min(r.start_s for r in rs), max(r.end_s for r in rs))
            for it, rs in sorted(by_iter.items())]


@dataclass(frozen=True)
class Marker:
    lane: str
    at_s: float
    kind: str  # llm_call | guardrail_rejected | guardrail_passed | delegation | tool_call
    label: str
    iteration: int = 0
    tokens: int = 0


def markers(trace: RunTrace) -> list[Marker]:
    """Point events for the timeline: LLM calls, guardrail verdicts, delegations."""
    origin = trace.origin
    spans = step_spans(trace)
    # The lane that did a step's work: the LangGraph node, or the CrewAI stage's role.
    lanes = {s.name: s.lane for s in spans if s.kind == "crew_task"}
    lanes |= {s.name: s.lane for s in spans if s.kind in ("node", "stage")}
    out: list[Marker] = []
    for e in trace.events:
        at = _rel(e.ended_at or e.started_at, origin)
        p = e.payload
        if e.event is TraceEventType.LLM_CALL:
            lane = str(p.get("role") or lanes.get(_core_step(e.agent, trace.mode), e.agent))
            label = (f"LLM call · {e.agent} · {p.get('schema') or p.get('model') or ''} · "
                     f"{e.tokens_in}+{e.tokens_out} tokens")
            out.append(Marker(lane, at, "llm_call", label, e.iteration,
                              e.tokens_in + e.tokens_out))
        elif e.event is TraceEventType.GUARDRAIL:
            passed = bool(p.get("passed", True))
            rules = ", ".join(p.get("rules") or []) or str(p.get("reason") or "")
            verdict = "passed" if passed else "rejected → retry"
            out.append(Marker(lanes.get(e.agent, e.agent), at,
                              "guardrail_passed" if passed else "guardrail_rejected",
                              f"guardrail {verdict} · {e.agent} {rules}".strip(), e.iteration))
        elif e.event is TraceEventType.DELEGATION:
            out.append(Marker(str(p.get("role") or e.agent), at, "delegation",
                              f"delegated to {p.get('coworker', '?')}", e.iteration))
    return out


def _core_step(agent: str, mode: PlannerMode) -> str:
    step = {"requirements": "parse_request", "attractions": "research_attractions",
            "accommodation": "research_accommodation", "dining": "research_dining",
            "budget": "allocate_budget", "composer": "compose_itinerary",
            "reviewer": "review"}.get(agent, agent)
    if step == "review" and mode is not PlannerMode.LANGGRAPH:
        return "review_itinerary"
    return step


# --------------------------------------------------------------------------- parallelism


@dataclass(frozen=True)
class Parallelism:
    """The research fan-out measured from timestamps (not assumed from the design)."""

    steps: int
    window_s: float  # first start → last end
    sum_s: float  # sum of the steps' own durations
    overlapping_pairs: int
    pairs: int

    @property
    def speedup(self) -> float | None:
        return round(self.sum_s / self.window_s, 2) if self.window_s > 0 else None

    @property
    def parallel(self) -> bool:
        return self.overlapping_pairs > 0

    @property
    def short(self) -> str:
        """Compact form for a metric tile, e.g. ``parallel ×4.8``."""
        if self.steps < 2:
            return "not measured"
        kind = ("parallel" if self.overlapping_pairs == self.pairs
                else "partial" if self.parallel else "sequential")
        return f"{kind} ×{self.speedup}" if self.speedup and kind != "sequential" else kind

    @property
    def label(self) -> str:
        if self.steps < 2:
            return "not measured"
        if self.overlapping_pairs == self.pairs:
            return f"parallel (all {self.steps} overlap)"
        if self.parallel:
            return f"partly parallel ({self.overlapping_pairs}/{self.pairs} pairs overlap)"
        return "sequential (no overlap)"


def parallelism(spans: Sequence[Span], names: Sequence[str] = RESEARCH_STEPS
                ) -> Parallelism | None:
    """Overlap of the research steps' spans; ``None`` when they were not recorded."""
    chosen: list[Span] = []
    for name in names:
        own = [s for s in spans if s.name == name and s.end is not None
               and s.status is StepStatus.DONE]
        # A CrewAI task span (async window) wins over its stage span; LangGraph has nodes.
        own = [s for s in own if s.kind in ("node", "crew_task")] or own
        if own:
            chosen.append(min(own, key=lambda s: s.start))
    if not chosen:
        return None
    start = min(s.start for s in chosen)
    end = max(s.end for s in chosen if s.end is not None)
    overlaps = sum(1 for a, b in combinations(chosen, 2)
                   if a.end is not None and b.end is not None
                   and a.start < b.end and b.start < a.end)
    return Parallelism(steps=len(chosen), window_s=round((end - start).total_seconds(), 3),
                       sum_s=round(sum(s.duration_s or 0 for s in chosen), 3),
                       overlapping_pairs=overlaps, pairs=len(chosen) * (len(chosen) - 1) // 2)


# --------------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class RunMetrics:
    """What the comparison view shows for one run. ``None`` = not recorded."""

    run_id: str
    mode: str
    status: str
    outcome: str | None
    duration_s: float | None
    llm_calls: int
    tokens_in: int
    tokens_out: int
    tokens: int
    llm_latency_s: float | None
    review_iterations: int | None
    guardrail_rejections: int | None
    delegations: int | None
    tool_calls: int | None
    steps: int | None
    research: Parallelism | None
    total_aed: float | None
    events: int


def _outcome(trace: RunTrace) -> str | None:
    if trace.itinerary is not None:
        return "approved" if trace.itinerary.approved else "needs revision"
    for e in reversed(trace.events):
        status = e.payload.get("status") if e.agent == "finalize" else None
        if status:
            return str(status).replace("_", " ")
    if trace.run.status is RunStatus.INTERRUPTED:
        return "needs clarification / interrupted"
    if trace.run.status is RunStatus.FAILED:
        return "failed"
    return None


def run_metrics(trace: RunTrace) -> RunMetrics:
    """Measurements from the run row and its events only."""
    run, events = trace.run, trace.events
    has = bool(events)
    spans = step_spans(trace)
    llm = [e for e in events if e.event is TraceEventType.LLM_CALL]
    iterations = run.iterations
    if iterations is None and has:
        iterations = max((e.iteration for e in events), default=0)
    latency = run.llm_latency_s if run.llm_latency_s else (
        round(sum(float(e.payload.get("latency_s") or 0) for e in llm), 3) if llm else None)
    duration = run.duration_s
    if duration is None and has and not trace.live:
        duration = round((max(e.ended_at or e.started_at for e in events)
                          - trace.origin).total_seconds(), 3)
    return RunMetrics(
        run_id=run.id, mode=run.mode.value, status=run.status.value, outcome=_outcome(trace),
        duration_s=None if duration is None else round(duration, 3),
        llm_calls=run.llm_calls or len(llm),
        tokens_in=run.tokens_in or sum(e.tokens_in for e in llm),
        tokens_out=run.tokens_out or sum(e.tokens_out for e in llm),
        tokens=run.tokens or sum(e.tokens_in + e.tokens_out for e in llm),
        llm_latency_s=latency, review_iterations=iterations,
        guardrail_rejections=sum(1 for e in events if e.event is TraceEventType.GUARDRAIL
                                 and not e.payload.get("passed", True)) if has else None,
        delegations=sum(e.event is TraceEventType.DELEGATION for e in events) if has else None,
        tool_calls=sum(e.event is TraceEventType.TOOL_CALL for e in events) if has else None,
        steps=len([s for s in spans if s.kind != "stage"]) if has else None,
        research=parallelism(spans) if has else None,
        total_aed=float(trace.itinerary.total_aed) if trace.itinerary else None,
        events=len(events))


@dataclass(frozen=True)
class ComparisonRow:
    metric: str
    a: Any
    b: Any
    delta: float | int | None  # b - a, for numbers only
    note: str = ""


_COMPARED: list[tuple[str, str, str]] = [
    ("outcome", "Outcome", ""),
    ("status", "Run status", ""),
    ("duration_s", "Wall-clock time (s)", "lower is faster"),
    ("llm_calls", "LLM calls", "agent-core + CrewAI agent calls"),
    ("tokens_in", "Prompt tokens", ""),
    ("tokens_out", "Completion tokens", ""),
    ("tokens", "Total tokens", ""),
    ("llm_latency_s", "Time inside LLM calls (s)", "sum over calls"),
    ("review_iterations", "Review iterations (refinements)", "0 = first draft approved"),
    ("guardrail_rejections", "Guardrail rejections", "CrewAI task retries"),
    ("delegations", "Delegations", "hierarchical manager"),
    ("tool_calls", "Tool calls", "CrewAI tool usage"),
    ("steps", "Orchestration steps", "nodes or tasks executed"),
    ("total_aed", "Itinerary total (AED, estimate)", ""),
]


def compare(a: RunMetrics, b: RunMetrics) -> list[ComparisonRow]:
    """Side-by-side rows; ``delta`` only when both sides were measured."""
    rows: list[ComparisonRow] = []
    for attr, label, note in _COMPARED:
        va, vb = getattr(a, attr), getattr(b, attr)
        delta: float | None = None
        if (isinstance(va, int | float) and isinstance(vb, int | float)
                and not isinstance(va, bool)):
            diff = round(float(vb) - float(va), 3) + 0.0  # + 0.0: no "-0.0"
            delta = int(diff) if isinstance(va, int) and isinstance(vb, int) else diff
        rows.append(ComparisonRow(label, va, vb, delta, note))
    pa, pb = a.research, b.research
    rows.append(ComparisonRow("Research execution", pa.label if pa else None,
                              pb.label if pb else None, None,
                              "measured overlap of the 5 research steps"))
    rows.append(ComparisonRow(
        "Research window (s)", pa.window_s if pa else None, pb.window_s if pb else None,
        round(pb.window_s - pa.window_s, 3) if pa and pb else None,
        "first research start → last research end"))
    rows.append(ComparisonRow(
        "Research speed-up (Σ durations / window)", pa.speedup if pa else None,
        pb.speedup if pb else None,
        round(pb.speedup - pa.speedup, 2) if pa and pb and pa.speedup and pb.speedup
        else None, "≈1 = sequential, >1 = parallel"))
    return rows


@dataclass(frozen=True)
class FairnessCheck:
    label: str
    same: bool | None  # None = cannot tell from what was recorded
    a: str
    b: str


def recorded_request(trace: RunTrace) -> dict[str, Any] | None:
    """The trip request the run planned, from its requirements step output."""
    for e in trace.events:
        if e.agent == "parse_request" and e.event is TraceEventType.END:
            req = (e.payload.get("output") or {}).get("request")
            if req:
                return dict(req)
    if trace.itinerary is not None and trace.itinerary.request is not None:
        return trace.itinerary.request.model_dump(mode="json")
    return None


def fairness(a: RunTrace, b: RunTrace) -> list[FairnessCheck]:
    """Is the comparison like for like? Request, dataset, model and loop cap."""
    def show(v: Any) -> str:
        return "not recorded" if v is None else str(v)

    def check(label: str, va: Any, vb: Any) -> FairnessCheck:
        same = None if va is None or vb is None else va == vb
        return FairnessCheck(label, same, show(va), show(vb))

    ra, rb = recorded_request(a), recorded_request(b)
    ca, cb = a.run.context, b.run.context
    return [
        check("Same user message", a.message, b.message),
        FairnessCheck("Same trip request", None if ra is None or rb is None else ra == rb,
                      _request_line(ra), _request_line(rb)),
        check("Same seed dataset (fingerprint)", ca.get("data"), cb.get("data")),
        check("Same LLM provider", ca.get("provider"), cb.get("provider")),
        check("Same model", ca.get("model"), cb.get("model")),
        check("Same review-iteration cap", ca.get("max_review_iterations"),
              cb.get("max_review_iterations")),
        check("Same data-source mode (offline)", ca.get("http_offline"), cb.get("http_offline")),
    ]


def _request_line(req: dict[str, Any] | None) -> str:
    if not req:
        return "not recorded"
    trav = req.get("travellers") or {}
    return (f"{req.get('start_date')} → {req.get('end_date')}, {trav.get('adults', '?')} "
            f"adult(s), AED {req.get('budget_aed')}")


# --------------------------------------------------------------------------- inspector


@lru_cache(maxsize=1)
def output_models() -> dict[str, Any]:
    """Recorded ``output_schema`` name → Pydantic model (or adapter for lists)."""
    import dubai_planner.models as models
    from dubai_planner.agents import (
        accommodation,
        attractions,
        dining,
        requirements,
        revision,
        weather_events,
    )

    reg: dict[str, Any] = {}
    for name in models.__all__:
        obj = getattr(models, name)
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            reg[obj.__name__] = obj
    for cls in (attractions.AttractionsResult, accommodation.AccommodationResult,
                dining.DiningResult, requirements.RequirementsResult,
                weather_events.WeatherEventsResult, revision.RevisionPlan):
        reg[cls.__name__] = cls
    for name, cls in list(reg.items()):
        reg[f"list[{name}]"] = TypeAdapter(list[cls])  # type: ignore[valid-type]
    reg["list[Any]"] = TypeAdapter(list[Any])  # e.g. an empty list of violations
    return reg


@dataclass(frozen=True)
class ValidatedOutput:
    schema: str
    valid: bool | None  # None = schema unknown or output not recorded in full
    data: Any
    error: str | None = None


def validate_output(payload: dict[str, Any]) -> ValidatedOutput | None:
    """Re-validate a recorded step output against its model (the inspector's proof)."""
    schema = payload.get("output_schema")
    if not schema:
        return None
    if "output" not in payload:
        return ValidatedOutput(str(schema), None, payload.get("output_summary"),
                               "output too large to record in full; summary shown")
    model = output_models().get(str(schema))
    data = payload["output"]
    if model is None:
        return ValidatedOutput(str(schema), None, data, "no registered model for this schema")
    try:
        value = (model.validate_python(data) if isinstance(model, TypeAdapter)
                 else model.model_validate(data))
    except ValidationError as exc:
        return ValidatedOutput(str(schema), False, data, str(exc)[:500])
    dumped = (model.dump_python(value, mode="json") if isinstance(model, TypeAdapter)
              else value.model_dump(mode="json"))
    return ValidatedOutput(str(schema), True, dumped)


@dataclass(frozen=True)
class Inspection:
    header: dict[str, Any]
    inputs: dict[str, Any] | None
    output: ValidatedOutput | None
    prompt: list[dict[str, Any]]
    response: str | None
    details: dict[str, Any]


_SPLIT = ("inputs", "output", "output_schema", "output_summary", "output_chars",
          "output_truncated", "prompt", "response_preview")


def event_label(i: int, event: TraceEvent, origin: datetime) -> str:
    kind = event.payload.get("kind") or ""
    tag = f" [{kind}]" if kind else ""
    return (f"#{i:03d} +{_rel(event.started_at, origin):7.2f}s  {event.agent} · "
            f"{event.event.value}{tag} · it {event.iteration}")


def inspect_event(event: TraceEvent, origin: datetime) -> Inspection:
    """A safe view of one event; secrets are redacted again before display."""
    clean = TraceEvent.model_validate_json(redact(event.model_dump_json()))
    p = clean.payload
    header = {"agent": clean.agent, "event": clean.event.value, "kind": p.get("kind"),
              "mode": clean.mode, "iteration": clean.iteration,
              "start (s)": _rel(clean.started_at, origin),
              "end (s)": _rel(clean.ended_at, origin) if clean.ended_at else None,
              "duration (s)": round(clean.duration_s, 3) if clean.duration_s is not None
              else None, "tokens in": clean.tokens_in, "tokens out": clean.tokens_out}
    if "latency_s" in p:
        header["LLM latency (s)"] = p["latency_s"]
    prompt = p.get("prompt")
    return Inspection(
        header=header, inputs=p.get("inputs"), output=validate_output(p),
        prompt=list(prompt) if isinstance(prompt, list) else [],
        response=p.get("response_preview"),
        details={k: v for k, v in p.items() if k not in _SPLIT})


def state_changes(trace: RunTrace) -> list[dict[str, Any]]:
    """The shared-state updates in order (LangGraph node updates, CrewAI task results)."""
    origin = trace.origin
    return [{"at (s)": _rel(e.started_at, origin), "step": e.agent, "iteration": e.iteration,
             "update": json.dumps(e.payload.get("update", {}), default=str)[:300]}
            for e in trace.events
            if e.event is TraceEventType.INFO and e.payload.get("kind") == "state_update"]


def recorded_attempts(trace: RunTrace) -> list[dict[str, Any]]:
    """Review attempts with their issues, rebuilt from the trace (survives restarts)."""
    out: list[dict[str, Any]] = []
    for e in trace.events:
        p = e.payload
        if e.event is TraceEventType.END and e.agent == "review" and p.get("kind") == "node":
            report = p.get("output") or {}
            out.append({"iteration": e.iteration, "attempt": p.get("attempt") or {},
                        "issues": [{"severity": i.get("severity"), "rule": i.get("rule_id"),
                                    "day": i.get("day_index"), "message": i.get("message")}
                                   for i in report.get("issues", [])]})
        elif e.event is TraceEventType.GUARDRAIL and p.get("attempt"):
            out.append({"iteration": e.iteration, "attempt": p["attempt"],
                        "issues": list(p.get("issues") or [])})
    return out


# --------------------------------------------------------------------------- graphs


@dataclass(frozen=True)
class GraphSpec:
    """A framework-neutral picture of an orchestration: nodes, edges, parallel group."""

    nodes: list[tuple[str, str]]  # (id, label)
    edges: list[tuple[str, str, bool, str]]  # (source, target, dashed, label)
    parallel: list[str]
    parallel_label: str
    title: str

    @property
    def steps(self) -> list[str]:
        return [n for n, _ in self.nodes if not n.startswith("__")]


def langgraph_spec(description: dict[str, Any]) -> GraphSpec:
    """From ``orchestration.langgraph.graph.describe_graph`` (the compiled graph)."""
    nodes = [(n, "START" if n == "__start__" else "END" if n == "__end__" else n)
             for n in description["nodes"]]
    edges = [(e["source"], e["target"], bool(e["conditional"]), "")
             for e in description["edges"]]
    return GraphSpec(nodes, edges, list(description.get("parallel", [])),
                     "parallel research (one super-step)", "LangGraph StateGraph")


def crewai_spec(description: dict[str, Any]) -> GraphSpec:
    """From ``CrewFactory.describe()`` (YAML roles and tasks, context dependencies)."""
    tasks = description["tasks"]
    nodes = [("__start__", "START")]
    nodes += [(t["name"], f"{t['name']}\n{t['role']}") for t in tasks]
    nodes.append(("__end__", "END"))
    edges: list[tuple[str, str, bool, str]] = [("__start__", "parse_request", False, "")]
    asynchronous = [t["name"] for t in tasks if t.get("async")]
    research = [t["name"] for t in tasks if t["name"].startswith("research_")]
    for name in research:
        edges.append(("parse_request", name, False, ""))
    for t in tasks:
        for dep in t.get("context", []):
            if t["name"] == "compose_itinerary" and dep in research:
                continue  # drawn through the budget task, as context flows
            edges.append((dep, t["name"], False, "context"))
    edges += [("compose_itinerary", "compose_itinerary", True, "guardrail retry"),
              ("review_itinerary", "compose_itinerary", True, "revision crew"),
              ("review_itinerary", "__end__", False, "")]
    manager = description.get("manager")
    if manager:
        nodes.insert(1, ("__manager__", f"manager\n{manager}"))
        edges += [("__manager__", t["name"], True, "") for t in tasks]
    label = ("async research tasks" if asynchronous
             else "research tasks (run one by one by the manager)")
    process = description.get("process", "sequential")
    return GraphSpec(nodes, edges, research, label, f"CrewAI crew ({process})")


_FILL = {StepStatus.PENDING: ("#f1f0ec", "#8a8985"), StepStatus.RUNNING: ("#fab219", "#0b0b0b"),
         StepStatus.DONE: ("#0ca30c", "#0b6b0b"), StepStatus.FAILED: ("#d03b3b", "#8f1f1f"),
         StepStatus.WAITING: ("#ec835a", "#8f4a2b"), StepStatus.STOPPED: ("#c3c2b7", "#52514e")}
_FONT = {StepStatus.PENDING: "#52514e", StepStatus.RUNNING: "#0b0b0b",
         StepStatus.DONE: "#ffffff", StepStatus.FAILED: "#ffffff",
         StepStatus.WAITING: "#0b0b0b", StepStatus.STOPPED: "#0b0b0b"}


def _q(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def to_dot(spec: GraphSpec, status: dict[str, StepStatus] | None = None) -> str:
    """Graphviz DOT with status colours *and* status words (never colour alone)."""
    status = status or {}
    lines = ["digraph G {", "  rankdir=TB; bgcolor=transparent; nodesep=0.25; ranksep=0.35;",
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, '
             'penwidth=1.2];', '  edge [color="#8a8985", fontname="Helvetica", fontsize=9];']

    def node(nid: str, label: str) -> str:
        if nid in ("__start__", "__end__"):
            return f'  {_q(nid)} [label={_q(label)}, shape=oval, fillcolor="#e1e0d9"];'
        st = status.get(nid, StepStatus.PENDING)
        fill, border = _FILL[st]
        width = 3 if st is StepStatus.RUNNING else 1.2
        text = f"{label}\n{STATUS_ICON[st]} {st.value}"
        return (f"  {_q(nid)} [label={_q(text)}, fillcolor={_q(fill)}, color={_q(border)}, "
                f"fontcolor={_q(_FONT[st])}, penwidth={width}];")

    labels = dict(spec.nodes)
    if spec.parallel:
        lines.append("  subgraph cluster_parallel {")
        lines.append(f'    label={_q(spec.parallel_label)}; style="dashed,rounded"; '
                     'color="#6250d6"; fontcolor="#6250d6"; fontsize=10;')
        lines += ["  " + node(n, labels.get(n, n)) for n in spec.parallel if n in labels]
        lines.append("  }")
    lines += [node(n, label) for n, label in spec.nodes if n not in spec.parallel]
    for src, dst, dashed, label in spec.edges:
        attrs = ['style=dashed'] if dashed else []
        if label:
            attrs.append(f"label={_q(label)}")
        lines.append(f"  {_q(src)} -> {_q(dst)}" + (f" [{', '.join(attrs)}]" if attrs else "")
                     + ";")
    lines.append("}")
    return "\n".join(lines)


def to_mermaid(spec: GraphSpec, status: dict[str, StepStatus] | None = None) -> str:
    """Mermaid flowchart with the same status classes (for reports and docs)."""
    status = status or {}

    def mid(n: str) -> str:
        return n.strip("_") or "node"

    lines = ["flowchart TD"]
    for n, label in spec.nodes:
        text = label.replace("\n", "<br/>").replace('"', "'")
        lines.append(f'    {mid(n)}["{text}"]')
    for src, dst, dashed, label in spec.edges:
        arrow = "-.->" if dashed else "-->"
        lines.append(f"    {mid(src)} {arrow}{'|' + label + '|' if label else ''} {mid(dst)}")
    for st, (fill, border) in _FILL.items():
        lines.append(f"    classDef {st.value} fill:{fill},stroke:{border},color:{_FONT[st]}")
    for n, _ in spec.nodes:
        if n in status:
            lines.append(f"    class {mid(n)} {status[n].value}")
    return "\n".join(lines)
