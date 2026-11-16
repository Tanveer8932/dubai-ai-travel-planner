"""Run the LangGraph planner for a session: streaming, persistence, errors, resume.

:class:`LangGraphPlanner` is what the UI (Phase 5) and the evaluation runner (Phase 7)
use. One session is one LangGraph thread (``thread_id = session_id``):

- a new message starts a new planning request on the thread (the conversation and the
  request draft carry over); when the thread is waiting at the clarification gate, the
  message is the answer (``Command(resume=...)``);
- :meth:`LangGraphPlanner.resume` continues an unfinished run from its last checkpoint
  (after an error, a cancelled stream or a killed process);
- the session, its messages, the run record and the final itinerary version are stored
  in SQLite (:class:`Repository`), so nothing depends on ``st.session_state``.

Errors never escape as tracebacks from :meth:`stream`: the final update carries a
:class:`PlanResult` with ``status=failed`` and a short, secret-free error, and the
checkpoint lets the run be resumed. Errors are not hidden either: the run record is
marked ``failed`` and nothing unapproved is ever reported as approved.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal, overload

from langgraph.types import Command, Durability, RetryPolicy

from dubai_planner.config import Settings, get_settings
from dubai_planner.models import (
    ChatMessage,
    ChatRole,
    PlannerMode,
    PlannerUpdate,
    PlanResult,
    PlanStatus,
    RunStatus,
    TraceEvent,
    TraceEventType,
)
from dubai_planner.orchestration.langgraph import nodes as n
from dubai_planner.orchestration.langgraph.checkpoint import open_checkpointer
from dubai_planner.orchestration.langgraph.graph import build_graph, recursion_limit
from dubai_planner.orchestration.langgraph.state import new_run_input
from dubai_planner.storage import Repository
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import bus

MAX_MESSAGE_CHARS = 4000
#: Persist every super-step before the next one starts. LangGraph's default ("async")
#: writes checkpoints in the background, so a killed process can lose the last step(s);
#: "sync" makes resume-after-crash exact (AC3.5) at a small latency cost.
DURABILITY: Durability = "sync"
END_NODE = "__end__"

_RUN_STATUS = {
    PlanStatus.APPROVED: RunStatus.SUCCEEDED,
    PlanStatus.NEEDS_REVISION: RunStatus.SUCCEEDED,
    PlanStatus.NEEDS_CLARIFICATION: RunStatus.INTERRUPTED,
    PlanStatus.FAILED: RunStatus.FAILED,
    PlanStatus.CANCELLED: RunStatus.INTERRUPTED,
}


def safe_error(exc: BaseException) -> str:
    """Short error for users/run records. Our errors never contain secret values."""
    return f"{type(exc).__name__}: {str(exc).strip()[:300]}"


@dataclass
class _RunStats:
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


def _summarize(update: Any) -> dict[str, Any]:
    """A small, JSON-friendly summary of a node's state update (for progress UIs)."""
    if not isinstance(update, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in update.items():
        if key == "messages":
            continue
        if key == "research" and isinstance(value, dict):
            out |= {f"research.{k}": len(v) if isinstance(v, list) else bool(v)
                    for k, v in value.items()}
        elif isinstance(value, list):
            out[key] = len(value)
        elif isinstance(value, str | int | float | bool) or value is None:
            out[key] = value
        elif hasattr(value, "cost"):  # Itinerary
            out[key] = f"{len(value.days)} days, AED {value.cost.total}"
        elif hasattr(value, "approved") and hasattr(value, "score"):  # ReviewReport
            out[key] = {"approved": value.approved, "score": value.score}
        else:
            out[key] = type(value).__name__
    return out


class LangGraphPlanner:
    """LangGraph-mode planning service for any number of sessions (threads)."""

    def __init__(self, *, checkpointer: Any, tools: Toolbox | None = None, llm: Any = None,
                 repo: Repository | None = None, agents: n.AgentSuite | None = None,
                 settings: Settings | None = None, retry: RetryPolicy | None = None,
                 max_review_iterations: int | None = None, persist: bool = True) -> None:
        self.settings = settings or get_settings()
        self.repo = (repo or Repository(settings=self.settings)) if persist else None
        self.deps = n.PlannerDeps(
            tools=tools or Toolbox.default(), llm=llm, agents=agents or n.AgentSuite(),
            repo=self.repo,
            max_review_iterations=(self.settings.max_review_iterations
                                   if max_review_iterations is None else max_review_iterations))
        self.graph = build_graph(self.deps, checkpointer=checkpointer, retry=retry)

    @classmethod
    @contextmanager
    def open(cls, checkpoint_path: Path | str | None = None, **kwargs: Any
             ) -> Iterator[LangGraphPlanner]:
        """Planner with a SQLite checkpointer (``CHECKPOINT_DB_PATH`` by default)."""
        with open_checkpointer(checkpoint_path, settings=kwargs.get("settings")) as saver:
            yield cls(checkpointer=saver, **kwargs)

    # ------------------------------------------------------------------ config / state
    def config(self, session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id},
                "recursion_limit": recursion_limit(self.deps.max_review_iterations,
                                                   self.deps.max_clarifications)}

    def snapshot(self, session_id: str) -> Any:
        return self.graph.get_state(self.config(session_id))

    def pending_question(self, session_id: str) -> str | None:
        """The clarification question the thread is waiting on, if any."""
        for intr in self.snapshot(session_id).interrupts:
            value = intr.value
            if isinstance(value, dict):
                return str(value.get("question") or n.DEFAULT_QUESTION)
        return None

    def result(self, session_id: str, *, error: str | None = None) -> PlanResult | None:
        """The session's current outcome, read from the checkpoint only.

        A run that stopped before ``finalize`` without a known error (cancelled stream,
        killed process) is ``cancelled``; :meth:`resume` continues it.
        """
        snap = self.snapshot(session_id)
        values: dict[str, Any] = dict(snap.values or {})
        if not values:
            return None
        question = self.pending_question(session_id)
        status: PlanStatus | None = values.get("status")
        if question is not None:
            status = PlanStatus.NEEDS_CLARIFICATION
        elif error is not None:
            status = PlanStatus.FAILED
        elif snap.next or status is None:  # unfinished: stopped mid-run
            status = PlanStatus.CANCELLED
        best = values.get("best")
        finished = not snap.next and question is None
        itinerary = values.get("itinerary") if finished else (best.itinerary if best else None)
        review = values.get("review") if finished else (best.review if best else None)
        violations = values.get("validation") if finished else (
            best.violations if best else [])
        return PlanResult(
            session_id=session_id, run_id=values.get("run_id", ""), mode=n.MODE,
            status=status, request=values.get("request"), draft=values.get("draft"),
            itinerary=itinerary, review=review, violations=list(violations or []),
            attempts=list(values.get("attempts", [])), iterations=values.get("iteration", 0),
            clarification_question=question or (
                values.get("clarification_question")
                if status is PlanStatus.NEEDS_CLARIFICATION else None),
            missing=list(values.get("missing", [])),
            assumptions=list(values.get("assumptions", [])),
            saved_version=values.get("saved_version"), error=error,
        )

    # ------------------------------------------------------------------ running
    def stream(self, message: str | None, session_id: str, *, today: date | None = None
               ) -> Iterator[PlannerUpdate]:
        """Plan (or answer the pending clarification) and yield one update per node.

        ``message=None`` resumes an unfinished run. The last update has ``result`` set.
        """
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        cfg = self.config(session_id)
        snap = self.graph.get_state(cfg)
        values: dict[str, Any] = dict(snap.values or {})
        graph_input: Any
        if message is None:
            if not snap.next:
                raise ValueError(f"session {session_id!r} has no unfinished run to resume")
            graph_input, run_id = None, values.get("run_id") or uuid.uuid4().hex
        else:
            text = message.strip()
            if not text or len(text) > MAX_MESSAGE_CHARS:
                raise ValueError(f"message must be 1-{MAX_MESSAGE_CHARS} characters")
            if snap.interrupts:
                graph_input, run_id = Command(resume=text), values["run_id"]
            else:
                run_id = uuid.uuid4().hex
                graph_input = new_run_input(
                    session_id=session_id, run_id=run_id, message=text,
                    today=today or date.today(),
                    user_message=ChatMessage(role=ChatRole.USER, content=text))
        self._record_start(session_id, run_id, message)
        stats = _RunStats(run_id)
        unsubscribe = bus.subscribe(stats)
        iteration = int(values.get("iteration", 0)) if graph_input is None or isinstance(
            graph_input, Command) else 0
        result: PlanResult | None = None
        try:
            try:
                for chunk in self.graph.stream(graph_input, cfg, stream_mode="updates",
                                               durability=DURABILITY):
                    for node, update in chunk.items():
                        if node.startswith("__"):  # __interrupt__
                            continue
                        if isinstance(update, dict) and "iteration" in update:
                            iteration = int(update["iteration"])
                        yield PlannerUpdate(session_id=session_id, run_id=run_id, node=node,
                                            iteration=iteration, summary=_summarize(update))
                result = self.result(session_id)
            except GeneratorExit:
                self._record_end(session_id, run_id, self.result(session_id), stats)
                raise
            except Exception as exc:  # reported in the result and the run record, not hidden
                result = self.result(session_id, error=safe_error(exc)) or PlanResult(
                    session_id=session_id, run_id=run_id, mode=n.MODE,
                    status=PlanStatus.FAILED, error=safe_error(exc))
        finally:
            unsubscribe()
        if result is None:  # pragma: no cover - every path above sets it
            raise RuntimeError("planner produced no result")
        self._record_end(session_id, run_id, result, stats)
        yield PlannerUpdate(session_id=session_id, run_id=run_id, node=END_NODE,
                            iteration=result.iterations, result=result)

    def run(self, message: str | None, session_id: str, *, today: date | None = None
            ) -> PlanResult:
        """Like :meth:`stream`, but only returns the final result."""
        final: PlanResult | None = None
        for update in self.stream(message, session_id, today=today):
            final = update.result or final
        if final is None:  # pragma: no cover - stream always ends with a result
            raise RuntimeError("planner produced no result")
        return final

    def resume(self, session_id: str) -> PlanResult:
        """Continue an unfinished run from its last checkpoint."""
        return self.run(None, session_id)

    # ------------------------------------------------------------------ persistence
    def _record_start(self, session_id: str, run_id: str, message: str | None) -> None:
        if self.repo is None:
            return
        if self.repo.get_session(session_id) is None:
            title = (message or "Dubai trip").strip().splitlines()[0][:60] or "Dubai trip"
            self.repo.create_session(title=title, mode=PlannerMode.LANGGRAPH,
                                     session_id=session_id)
        if message is not None:
            self.repo.add_message(session_id, ChatMessage(role=ChatRole.USER, content=message,
                                                          metadata={"run_id": run_id}))
        if self.repo.get_run(run_id) is None:
            self.repo.start_run(PlannerMode.LANGGRAPH, session_id=session_id, run_id=run_id)

    def _record_end(self, session_id: str, run_id: str, result: PlanResult | None,
                    stats: _RunStats) -> None:
        if self.repo is None or result is None:
            return
        prior = self.repo.get_run(run_id)
        calls = stats.llm_calls + (prior.llm_calls if prior else 0)
        tokens = stats.tokens + (prior.tokens if prior else 0)
        self.repo.finish_run(run_id, _RUN_STATUS[result.status], llm_calls=calls,
                             tokens=tokens, error=result.error)
        reply = self._reply(session_id, result)
        if reply:
            self.repo.add_message(session_id, ChatMessage(
                role=ChatRole.ASSISTANT, content=reply,
                metadata={"run_id": run_id, "status": result.status.value}))

    def _reply(self, session_id: str, result: PlanResult) -> str | None:
        if result.status is PlanStatus.NEEDS_CLARIFICATION and result.clarification_question:
            return result.clarification_question
        if result.status in (PlanStatus.APPROVED, PlanStatus.NEEDS_REVISION):
            messages = self.snapshot(session_id).values.get("messages", [])
            for msg in reversed(messages):
                if msg.role is ChatRole.ASSISTANT and msg.metadata.get("kind") == "plan_result":
                    return str(msg.content)
        return None


# --------------------------------------------------------------------------- function API


@overload
def run_langgraph(request_text: str | None, session_id: str, *, stream: Literal[True] = ...,
                  planner: LangGraphPlanner | None = ..., today: date | None = ...
                  ) -> Iterator[PlannerUpdate]: ...
@overload
def run_langgraph(request_text: str | None, session_id: str, *, stream: Literal[False],
                  planner: LangGraphPlanner | None = ..., today: date | None = ...
                  ) -> PlanResult: ...
def run_langgraph(request_text: str | None, session_id: str, *, stream: bool = True,
                  planner: LangGraphPlanner | None = None, today: date | None = None
                  ) -> Iterator[PlannerUpdate] | PlanResult:
    """Plan ``request_text`` for ``session_id`` (implementation plan, Phase 3 task 5).

    ``stream=True`` yields :class:`PlannerUpdate`s (the last one carries the result);
    ``stream=False`` returns the :class:`PlanResult`. Without ``planner`` a default one
    (settings-configured provider, ``CHECKPOINT_DB_PATH``, ``DB_PATH``) is opened.
    """
    if not stream:
        if planner is not None:
            return planner.run(request_text, session_id, today=today)
        with LangGraphPlanner.open() as default:
            return default.run(request_text, session_id, today=today)
    return _stream(request_text, session_id, planner, today)


def _stream(request_text: str | None, session_id: str, planner: LangGraphPlanner | None,
            today: date | None) -> Iterator[PlannerUpdate]:
    if planner is not None:
        yield from planner.stream(request_text, session_id, today=today)
        return
    with LangGraphPlanner.open() as default:
        yield from default.stream(request_text, session_id, today=today)
