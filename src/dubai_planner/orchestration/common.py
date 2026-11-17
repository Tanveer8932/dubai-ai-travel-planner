"""Framework-free helpers shared by the LangGraph and CrewAI planning services.

Both services report the same :class:`PlanResult`, write the same run records and
reply with the same assistant text, so these pieces live here rather than in either
adapter (hard rule 3: thin adapters, no duplicated logic).
"""

from __future__ import annotations

import hashlib
import logging
import random
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from dubai_planner.config import Settings
from dubai_planner.llm.errors import is_transient
from dubai_planner.models import (
    Attempt,
    AttemptSummary,
    PlanStatus,
    RunRecord,
    RunStatus,
    Severity,
    TraceEvent,
    TraceEventType,
)
from dubai_planner.redaction import MASK, redact, secret_values
from dubai_planner.storage import Repository
from dubai_planner.tracing import emit

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_MESSAGE_CHARS = 4000
DEFAULT_QUESTION = "Could you tell me your travel dates and total budget?"
END_NODE = "__end__"

#: How a plan outcome is recorded on the ``runs`` row.
RUN_STATUS = {
    PlanStatus.APPROVED: RunStatus.SUCCEEDED,
    PlanStatus.NEEDS_REVISION: RunStatus.SUCCEEDED,
    PlanStatus.NEEDS_CLARIFICATION: RunStatus.INTERRUPTED,
    PlanStatus.FAILED: RunStatus.FAILED,
    PlanStatus.CANCELLED: RunStatus.INTERRUPTED,
}


def safe_error(exc: BaseException) -> str:
    """Short error for users/run records, with any configured key value redacted.

    Our own errors never contain secrets; a provider SDK echoing a key is masked here.
    """
    return redact(f"{type(exc).__name__}: {str(exc).strip()[:300]}")


def check_message(message: str) -> str:
    text = message.strip()
    if not text or len(text) > MAX_MESSAGE_CHARS:
        raise ValueError(f"message must be 1-{MAX_MESSAGE_CHARS} characters")
    return text


#: Span kinds whose start/end is a step transition (node, CrewAI task/stage, crew). The
#: recorder writes to SQLite at each of them, so a live view polling SQLite sees every
#: transition (AC6.3) without one write per event.
TRANSITION_KINDS = frozenset({"node", "stage", "crew_task", "crew"})
_FLUSH_EVERY = 50


@dataclass
class RunStats:
    """Bus subscriber for one run: counts LLM usage and persists the run's trace events.

    With a ``repo`` the events are written to ``trace_events`` (secrets redacted) at
    every step transition, every guardrail verdict and when the run ends
    (:meth:`flush`), so traces survive an app restart (AC6.5) and a live view can follow
    the run from SQLite alone. A failing write is logged and never stops planning.
    """

    run_id: str
    repo: Repository | None = None
    settings: Settings | None = None
    llm_calls: int = 0
    tokens: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_latency_s: float = 0.0
    events: list[TraceEvent] = field(default_factory=list)
    _pending: list[TraceEvent] = field(default_factory=list, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __call__(self, event: TraceEvent) -> None:
        if event.run_id != self.run_id:
            return
        with self._lock:
            self.events.append(event)
            if event.event is TraceEventType.LLM_CALL:
                self.llm_calls += 1
                self.tokens_in += event.tokens_in
                self.tokens_out += event.tokens_out
                self.tokens += event.tokens_in + event.tokens_out
                latency = event.payload.get("latency_s")
                if isinstance(latency, int | float):
                    self.llm_latency_s += float(latency)
                elif event.duration_s:
                    self.llm_latency_s += event.duration_s
            if self.repo is None:
                return
            self._pending.append(event)
            transition = (event.event in (TraceEventType.START, TraceEventType.END,
                                          TraceEventType.ERROR)
                          and event.payload.get("kind") in TRANSITION_KINDS)
            if (transition or event.event is TraceEventType.GUARDRAIL
                    or len(self._pending) >= _FLUSH_EVERY):
                self.flush()

    def flush(self) -> int:
        """Write pending events to SQLite (in publish order); returns how many."""
        with self._lock:
            if self.repo is None or not self._pending:
                return 0
            batch, self._pending = self._pending, []
            try:
                self.repo.add_trace_events([redact_event(e, self.settings) for e in batch])
            except (sqlite3.Error, OSError):  # observability must not break planning
                logger.warning("could not persist %d trace event(s) of run %s",
                               len(batch), self.run_id, exc_info=True)
                return 0
            return len(batch)

    def finish_kwargs(self, prior: RunRecord | None) -> dict[str, Any]:
        """Aggregates for ``Repository.finish_run`` (added to a resumed run's totals)."""
        def plus(name: str) -> Any:
            return getattr(self, name) + (getattr(prior, name) if prior else 0)

        return {"llm_calls": plus("llm_calls"), "tokens": plus("tokens"),
                "tokens_in": plus("tokens_in"), "tokens_out": plus("tokens_out"),
                "llm_latency_s": plus("llm_latency_s")}


def redact_event(event: TraceEvent, settings: Settings | None = None) -> TraceEvent:
    """``event`` with any configured secret value removed from its payload."""
    values = secret_values(settings)
    if not values:
        return event
    raw = event.model_dump_json()
    clean = raw
    for value in values:
        clean = clean.replace(value, MASK)
    return event if clean == raw else TraceEvent.model_validate_json(clean)


@lru_cache(maxsize=8)
def data_fingerprint(seed_dir: str) -> str:
    """Short hash of the seed JSON files: equal fingerprints = same dataset."""
    digest = hashlib.sha256()
    root = Path(seed_dir)
    for path in sorted(root.glob("*.json")) if root.is_dir() else []:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def run_context_info(settings: Settings, **extra: Any) -> dict[str, Any]:
    """What a fair comparison of two runs needs to know. Never contains secrets."""
    return {"provider": settings.llm_provider.value, "model": settings.model_for(None),
            "data": data_fingerprint(str(settings.seed_dir)),
            "http_offline": settings.http_offline,
            "max_review_iterations": settings.max_review_iterations,
            "budget_tolerance": str(settings.budget_tolerance), **extra}


def summary_text(status: PlanStatus, attempt: Attempt | None, question: str | None,
                 iterations: int) -> str:
    """Assistant reply stored in the conversation when a run ends."""
    if status is PlanStatus.NEEDS_CLARIFICATION:
        return question or DEFAULT_QUESTION
    if attempt is None:
        return "I could not produce an itinerary."
    it = attempt.itinerary
    head = (f"{it.title or 'Your Dubai itinerary'}: {len(it.days)} day(s), estimated total "
            f"AED {it.cost.total} (budget AED {it.cost.budget_aed}).")
    if status is PlanStatus.APPROVED:
        return f"{head} Approved after {iterations} revision(s)."
    blocking = [i.message for i in attempt.review.blocking_issues][:3]
    return (f"{head} Not approved after {iterations} revision(s); unresolved: "
            + "; ".join(blocking))


def attempt_summary(attempt: Attempt, locked_days: list[int]) -> AttemptSummary:
    """The compact per-attempt record both modes put on :class:`PlanResult`."""
    violations = attempt.violations
    errors = [v for v in violations if v.severity is Severity.ERROR]
    return AttemptSummary(
        iteration=attempt.iteration, approved=attempt.approved, score=attempt.review.score,
        total_aed=attempt.itinerary.cost.total, errors=attempt.errors,
        warnings=sum(v.severity is Severity.WARNING for v in violations),
        blocking_rules=sorted({v.rule_id for v in errors}), locked_days=list(locked_days))


def better(candidate: Attempt, best: Attempt | None) -> Attempt:
    """Keep the best attempt: approved, then fewer errors, higher score, later."""
    return candidate if best is None or candidate.rank() > best.rank() else best


@dataclass(frozen=True)
class RetrySettings:
    """Bounded retry for transient provider errors (on top of the SDK's own retries).

    Defaults match LangGraph mode's node ``RetryPolicy``: 4 attempts, waiting ≈ 8, 16,
    32 s, about the one-minute window of free-tier per-minute token limits.
    """

    max_attempts: int = 4
    initial_interval: float = 8.0
    backoff_factor: float = 2.0
    max_interval: float = 60.0
    jitter: bool = True


def call_with_retry(fn: Callable[[], T], retry: RetrySettings, *,
                    retry_on: Callable[[BaseException], bool] = is_transient,
                    sleep: Callable[[float], None] = time.sleep, agent: str = "retry") -> T:
    """Run ``fn``; retry only errors ``retry_on`` accepts, at most ``max_attempts`` times."""
    attempt = 1
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= retry.max_attempts or not retry_on(exc):
                raise
            wait = min(retry.max_interval,
                       retry.initial_interval * retry.backoff_factor ** (attempt - 1))
            if retry.jitter:
                wait *= 1 + random.random() * 0.25  # noqa: S311 - backoff jitter, not crypto
            emit(agent, TraceEventType.INFO, payload={
                "kind": "retry", "attempt": attempt, "wait_s": round(wait, 2),
                "error": type(exc).__name__})
            sleep(wait)
            attempt += 1


def summarize_value(value: Any) -> Any:
    """A small, JSON-friendly summary of one output value (for progress UIs)."""
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if hasattr(value, "cost") and hasattr(value, "days"):  # Itinerary
        return f"{len(value.days)} days, AED {value.cost.total}"
    if hasattr(value, "approved") and hasattr(value, "score"):  # ReviewReport
        return {"approved": value.approved, "score": value.score}
    return type(value).__name__
