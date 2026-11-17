"""In-process trace event bus shared by both orchestration modes.

Agents and the structured-output helper publish :class:`TraceEvent`s here; subscribers
(the SQLite writer and the Streamlit live view arrive in Phase 6, tests use
:func:`capture`) receive every event. The current run (``run_id``, ``mode``,
``iteration``) is carried in a ``ContextVar`` so it follows threads started with
``contextvars.copy_context`` and asyncio tasks, which is how LangGraph and CrewAI run
parallel branches.

Payloads must never contain secrets: only agent names, schema names, token counts,
latencies and short error summaries are published.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from dubai_planner.models.trace import TraceEvent, TraceEventType

logger = logging.getLogger(__name__)

Subscriber = Callable[[TraceEvent], None]


@dataclass(frozen=True)
class RunContext:
    run_id: str
    mode: str = "standalone"
    iteration: int = 0


_current_run: ContextVar[RunContext | None] = ContextVar("dubai_planner_run", default=None)


class TraceBus:
    """Thread-safe publish/subscribe. A failing subscriber never breaks the pipeline."""

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._lock = threading.Lock()

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        """Register ``subscriber``; returns a function that unsubscribes it."""
        with self._lock:
            self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            with self._lock:
                if subscriber in self._subscribers:
                    self._subscribers.remove(subscriber)

        return unsubscribe

    def publish(self, event: TraceEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber(event)
            except Exception:  # a broken UI/DB sink must not stop planning
                logger.exception("trace subscriber failed")


bus = TraceBus()


def current_run() -> RunContext:
    """The active run, or an ad-hoc one when an agent is called outside a run."""
    return _current_run.get() or RunContext(run_id="adhoc")


@contextmanager
def run_context(
    run_id: str | None = None, *, mode: str = "standalone", iteration: int = 0
) -> Iterator[RunContext]:
    ctx = RunContext(run_id=run_id or uuid.uuid4().hex, mode=mode, iteration=iteration)
    token = _current_run.set(ctx)
    try:
        yield ctx
    finally:
        _current_run.reset(token)


@contextmanager
def iteration_context(iteration: int) -> Iterator[RunContext]:
    """Same run, new review iteration (used by refine loops)."""
    ctx = replace(current_run(), iteration=iteration)
    token = _current_run.set(ctx)
    try:
        yield ctx
    finally:
        _current_run.reset(token)


def emit(
    agent: str,
    event: TraceEventType,
    *,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    payload: dict[str, Any] | None = None,
) -> TraceEvent:
    ctx = current_run()
    trace = TraceEvent(
        run_id=ctx.run_id,
        mode=ctx.mode,
        agent=agent,
        event=event,
        started_at=started_at or datetime.now(UTC),
        ended_at=ended_at,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        iteration=ctx.iteration,
        payload=payload or {},
    )
    bus.publish(trace)
    return trace


@contextmanager
def agent_span(agent: str, **payload: Any) -> Iterator[dict[str, Any]]:
    """Emit START, then END (or ERROR) with timing. Yields a dict for result payload."""
    started = datetime.now(UTC)
    emit(agent, TraceEventType.START, started_at=started, payload=dict(payload))
    result: dict[str, Any] = {}
    try:
        yield result
    except Exception as exc:
        emit(
            agent,
            TraceEventType.ERROR,
            started_at=started,
            ended_at=datetime.now(UTC),
            # Same ``kind`` as the start event, so viewers can pair them (Phase 6).
            payload={**({"kind": kind} if (kind := result.get("kind", payload.get("kind")))
                        else {}),
                     "error": type(exc).__name__, "message": str(exc)[:500]},
        )
        raise
    emit(agent, TraceEventType.END, started_at=started, ended_at=datetime.now(UTC),
         payload=result)


@contextmanager
def capture() -> Iterator[list[TraceEvent]]:
    """Collect every event published inside the block (tests and evaluation)."""
    events: list[TraceEvent] = []
    unsubscribe = bus.subscribe(events.append)
    try:
        yield events
    finally:
        unsubscribe()
