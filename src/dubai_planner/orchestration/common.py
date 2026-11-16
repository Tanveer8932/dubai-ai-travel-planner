"""Framework-free helpers shared by the LangGraph and CrewAI planning services.

Both services report the same :class:`PlanResult`, write the same run records and
reply with the same assistant text, so these pieces live here rather than in either
adapter (hard rule 3: thin adapters, no duplicated logic).
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from dubai_planner.llm.errors import is_transient
from dubai_planner.models import (
    Attempt,
    AttemptSummary,
    PlanStatus,
    RunStatus,
    Severity,
    TraceEvent,
    TraceEventType,
)
from dubai_planner.tracing import emit

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
    """Short error for users/run records. Our errors never contain secret values."""
    return f"{type(exc).__name__}: {str(exc).strip()[:300]}"


def check_message(message: str) -> str:
    text = message.strip()
    if not text or len(text) > MAX_MESSAGE_CHARS:
        raise ValueError(f"message must be 1-{MAX_MESSAGE_CHARS} characters")
    return text


@dataclass
class RunStats:
    """Bus subscriber counting one run's LLM calls and tokens (for the ``runs`` row)."""

    run_id: str
    llm_calls: int = 0
    tokens: int = 0
    events: list[TraceEvent] = field(default_factory=list)

    def __call__(self, event: TraceEvent) -> None:
        if event.run_id != self.run_id:
            return
        self.events.append(event)
        if event.event is TraceEventType.LLM_CALL:
            self.llm_calls += 1
            self.tokens += event.tokens_in + event.tokens_out


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
