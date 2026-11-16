"""Run the CrewAI planner for a session (implementation plan, Phase 4 task 7).

:class:`CrewAIPlanner` has the same service API and returns the same
:class:`PlanResult` / :class:`PlannerUpdate` contracts as LangGraph mode, so the UI
(Phase 5) and the evaluation runner (Phase 7) never branch on the framework.

One planning request:

1. **Requirements crew** (Requirements Analyst). If the request is incomplete the run
   ends ``needs_clarification`` with the agent's question; the draft is stored with
   the question in SQLite and the next message on the session answers it (same
   ``run_id``), up to 3 rounds, like LangGraph's ``interrupt``.
2. **Planning crew**: research (concurrent) → budget → compose (rule guardrail: a
   hard-rule error is rejected with feedback and CrewAI retries the task) → review.
3. **Outer review loop**: if the reviewed attempt is not approved and refinements are
   left, the shared ``plan_revision`` turns the review into targeted feedback and a
   **revision crew** (compose → review) is kicked off. Guardrail retries and outer
   iterations share one budget, ``MAX_REVIEW_ITERATIONS``, so at most
   ``1 + MAX_REVIEW_ITERATIONS`` itineraries are composed.
4. The best attempt is returned (approved only if the reviewer approved **and** no
   error-severity rule fails) and saved as the session's next itinerary version.

Session context lives in SQLite only (messages, run records, itinerary versions; the
request draft in message metadata). CrewAI has no checkpointer here: :meth:`resume`
re-runs an unfinished request from its start rather than from the failed step.
Errors never escape :meth:`stream` as tracebacks: the final update carries a ``failed``
result with a short, secret-free error, and the run record is marked ``failed``.
"""

from __future__ import annotations

import contextvars
import queue
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, overload

from crewai.llms.base_llm import BaseLLM
from crewai.tasks.task_output import TaskOutput

from dubai_planner.agents.revision import plan_revision
from dubai_planner.agents.suite import AgentSuite
from dubai_planner.config import Settings, get_settings
from dubai_planner.llm import as_chat_llm, get_crewai_llm
from dubai_planner.llm.mock import MockLLM
from dubai_planner.models import (
    ChatMessage,
    ChatRole,
    PlannerMode,
    PlannerUpdate,
    PlanResult,
    PlanStatus,
    TraceEventType,
    TripRequestDraft,
)
from dubai_planner.orchestration.common import (
    DEFAULT_QUESTION,
    END_NODE,
    RUN_STATUS,
    RetrySettings,
    RunStats,
    check_message,
    safe_error,
    summarize_value,
    summary_text,
)
from dubai_planner.orchestration.crewai.board import (
    STAGES,
    CrewRunAborted,
    CrewRunBoard,
    CrewRunCancelled,
)
from dubai_planner.orchestration.crewai.crew import CrewDeps, CrewFactory, CrewProcessName
from dubai_planner.orchestration.crewai.listener import trace_listener
from dubai_planner.orchestration.crewai.stages import StageRunner
from dubai_planner.storage import Repository
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import agent_span, bus, emit, iteration_context, run_context

MAX_CLARIFICATIONS = 3
_TASK_TO_STAGE = {s.task: s.key for s in STAGES.values()}
_MODES: dict[str, PlannerMode] = {"sequential": PlannerMode.CREWAI_SEQUENTIAL,
                                  "hierarchical": PlannerMode.CREWAI_HIERARCHICAL}


@dataclass
class _Pending:
    """A clarification the session is waiting on (from the last assistant message)."""

    run_id: str
    draft: TripRequestDraft | None
    rounds: int
    question: str


class _LLMs:
    """CrewAI LLM per role (agent reasoning) and chat client per agent (agent core)."""

    def __init__(self, llm: Any, settings: Settings) -> None:
        self.llm, self.settings = llm, settings
        self._core: dict[str, Any] = {}
        self._lock = threading.Lock()

    def crew_llm(self, name: str) -> Any:
        """A fresh CrewAI LLM (agents mutate theirs, so they are never shared)."""
        if isinstance(self.llm, MockLLM):
            from dubai_planner.llm.crewai_mock import MockCrewAILLM

            return MockCrewAILLM(engine=self.llm)
        if isinstance(self.llm, BaseLLM):
            return self.llm.model_copy()
        if self.llm is not None:
            raise TypeError("CrewAI mode needs a CrewAI LLM or the mock engine as `llm`")
        return get_crewai_llm(name, self.settings)

    def core(self, agent: str) -> Any:
        """Chat client for an agent-core call: the CrewAI LLM behind a neutral adapter."""
        with self._lock:
            if agent not in self._core:
                self._core[agent] = as_chat_llm(self.crew_llm(agent))
            return self._core[agent]


class CrewAIPlanner:
    """CrewAI-mode planning service for any number of sessions."""

    def __init__(self, *, process: CrewProcessName | None = None, tools: Toolbox | None = None,
                 llm: Any = None, repo: Repository | None = None,
                 agents: AgentSuite | None = None, settings: Settings | None = None,
                 retry: RetrySettings | None = None, max_review_iterations: int | None = None,
                 persist: bool = True) -> None:
        self.settings = settings or get_settings()
        self.process: CrewProcessName = process or self.settings.crewai_process
        if self.process not in _MODES:
            raise ValueError(f"unknown CrewAI process {self.process!r}")
        self.mode = _MODES[self.process]
        self.repo = (repo or Repository(settings=self.settings)) if persist else None
        self.tools = tools or Toolbox.default()
        self.max_review_iterations = (self.settings.max_review_iterations
                                      if max_review_iterations is None else max_review_iterations)
        llms = _LLMs(llm, self.settings)
        self.factory = CrewFactory(CrewDeps(
            runner=StageRunner(tools=self.tools, llm_for=llms.core,
                               agents=agents or AgentSuite()),
            crew_llm_for=lambda role_key: llms.crew_llm(
                STAGES_BY_ROLE.get(role_key) or role_key),
            process=self.process, max_iter=self.settings.crewai_agent_max_iter,
            retry=retry or RetrySettings()))

    # ------------------------------------------------------------------ session context
    def _history(self, session_id: str) -> list[ChatMessage]:
        return self.repo.chat_history(session_id) if self.repo is not None else []

    def _last_assistant(self, session_id: str) -> ChatMessage | None:
        for message in reversed(self._history(session_id)):
            if message.role is ChatRole.ASSISTANT:
                return message
        return None

    def pending_question(self, session_id: str) -> str | None:
        pending = self._pending(session_id)
        return pending.question if pending else None

    def _pending(self, session_id: str) -> _Pending | None:
        last = self._last_assistant(session_id)
        if last is None or last.metadata.get("kind") != "clarification":
            return None
        if last.metadata.get("gave_up"):
            return None
        draft = last.metadata.get("draft")
        return _Pending(run_id=str(last.metadata.get("run_id") or uuid.uuid4().hex),
                        draft=TripRequestDraft.model_validate(draft) if draft else None,
                        rounds=int(last.metadata.get("clarifications", 0)),
                        question=last.content)

    def _previous_draft(self, session_id: str) -> TripRequestDraft | None:
        """The latest request draft of the session (refinements build on it)."""
        for message in reversed(self._history(session_id)):
            draft = message.metadata.get("draft") if message.role is ChatRole.ASSISTANT else None
            if draft:
                return TripRequestDraft.model_validate(draft)
        return None

    def _unfinished(self, session_id: str) -> tuple[str, str] | None:
        """``(run_id, message)`` of the latest run that did not finish, for resume."""
        if self.repo is None:
            return None
        runs = [r for r in self.repo.list_runs(session_id) if r.mode is self.mode]
        if not runs or runs[-1].status.value not in ("failed", "interrupted", "running"):
            return None
        run = runs[-1]
        for message in reversed(self._history(session_id)):
            if message.role is ChatRole.USER and message.metadata.get("run_id") == run.id:
                return run.id, message.content
        return None

    # ------------------------------------------------------------------ running
    def stream(self, message: str | None, session_id: str, *, today: date | None = None
               ) -> Iterator[PlannerUpdate]:
        """Plan (or answer the pending clarification); yield one update per finished task.

        ``message=None`` re-runs the session's unfinished request from the start. The
        last update has ``result`` set.
        """
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        record_message = True
        if message is None:
            unfinished = self._unfinished(session_id)
            if unfinished is None:
                raise ValueError(f"session {session_id!r} has no unfinished run to resume")
            run_id, text = unfinished
            record_message, pending = False, None
        else:
            text = check_message(message)
            pending = self._pending(session_id)
            run_id = pending.run_id if pending else uuid.uuid4().hex
        history = self._history(session_id)
        self._record_start(session_id, run_id, text if record_message else None)
        board = CrewRunBoard(
            session_id=session_id, run_id=run_id, mode=self.mode, today=today or date.today(),
            user_message=text, history=history,
            previous_draft=pending.draft if pending else self._previous_draft(session_id),
            max_iterations=self.max_review_iterations)
        rounds = pending.rounds if pending else 0
        stats = RunStats(run_id)
        updates: queue.Queue[PlannerUpdate | BaseException | None] = queue.Queue()
        outcome: dict[str, Any] = {}
        ctx = contextvars.copy_context()
        worker = threading.Thread(
            target=ctx.run, args=(self._work, board, rounds, updates, outcome),
            name=f"crewai-{run_id[:8]}", daemon=True)
        unsubscribe = bus.subscribe(stats)
        result: PlanResult | None = None
        try:
            worker.start()
            try:
                while True:
                    item = updates.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):  # pragma: no cover - _work catches
                        raise item
                    yield item
            except GeneratorExit:
                board.cancelled = True  # the crew stops at its next tool/guardrail
                worker.join(timeout=120)
                done = outcome.get("result") or PlanResult(
                    session_id=session_id, run_id=run_id, mode=self.mode,
                    status=PlanStatus.CANCELLED, iterations=board.iteration)
                self._finish(board, done, stats)
                raise
            worker.join()
            result = outcome["result"]
        finally:
            trace_listener().unregister(board)
            unsubscribe()
        if result is None:  # pragma: no cover - _work always sets a result
            raise RuntimeError("planner produced no result")
        self._finish(board, result, stats)
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
        """Re-run the session's unfinished request (CrewAI mode has no checkpoints)."""
        return self.run(None, session_id)

    # ------------------------------------------------------------------ the run itself
    def _work(self, board: CrewRunBoard, rounds: int,
              updates: queue.Queue[PlannerUpdate | BaseException | None],
              outcome: dict[str, Any]) -> None:
        def on_task(output: TaskOutput) -> None:
            stage = _TASK_TO_STAGE.get(str(output.name or ""), "")
            value = output.pydantic
            summary: dict[str, Any] = {stage or str(output.name): summarize_value(value)}
            updates.put(PlannerUpdate(session_id=board.session_id, run_id=board.run_id,
                                      node=str(output.name or stage), iteration=board.iteration,
                                      summary=summary))

        try:
            with run_context(board.run_id, mode=self.mode.value):
                outcome["result"] = self._plan(board, rounds, on_task)
        except CrewRunCancelled:
            outcome["result"] = self._result(board, PlanStatus.CANCELLED)
        except Exception as exc:  # reported in the result and the run record, not hidden
            cause = board.error if board.error is not None else exc
            outcome["result"] = self._result(board, PlanStatus.FAILED, error=safe_error(cause))
        finally:
            trace_listener().flush()
            updates.put(None)

    def _kickoff(self, built: Any, board: CrewRunBoard, name: str) -> None:
        with iteration_context(board.iteration), agent_span(
                name, kind="crew", process=self.process, tasks=list(built.tasks)):
            try:
                built.kickoff(board)
            except Exception as exc:
                if board.cancelled:
                    raise CrewRunCancelled("the planning run was cancelled") from exc
                if board.error is not None:
                    raise CrewRunAborted(f"stage {board.failed_stage!r} failed") from board.error
                raise
            finally:
                trace_listener().flush()
        if board.error is not None:  # pragma: no cover - guardrails re-raise it earlier
            raise CrewRunAborted(f"stage {board.failed_stage!r} failed") from board.error

    def _plan(self, board: CrewRunBoard, rounds: int,
              on_task: Callable[[TaskOutput], None]) -> PlanResult:
        self._kickoff(self.factory.requirements_crew(board, on_task), board, "requirements_crew")
        req = board.requirements
        if req is None:  # pragma: no cover - the guardrail guarantees it
            raise RuntimeError("the requirements task produced no result")
        if req.request is None:
            gave_up = rounds + 1 >= MAX_CLARIFICATIONS
            question = req.clarification_question or DEFAULT_QUESTION
            emit("clarification_gate", TraceEventType.INFO, payload={
                "kind": "gate", "complete": False, "missing": req.missing,
                "round": rounds + 1, "gave_up": gave_up})
            return self._result(board, PlanStatus.NEEDS_CLARIFICATION, question=question,
                                rounds=rounds + 1, gave_up=gave_up)
        self._kickoff(self.factory.planning_crew(board, on_task), board, "planning_crew")
        while True:
            last = board.last_attempt
            if last is None:  # pragma: no cover - the review guardrail records one
                raise RuntimeError("the review task recorded no attempt")
            if last.approved or not board.can_revise:
                break
            plan = plan_revision(last.review, last.itinerary, num_days=board.request.num_days)
            emit("review_loop", TraceEventType.INFO, payload={
                "kind": "refine", "next_iteration": board.iteration + 1,
                "target_days": plan.target_days, "locked_days": plan.locked_days})
            board.start_revision(plan)
            self._kickoff(self.factory.revision_crew(board, on_task), board, "revision_crew")
        best = board.best
        status = PlanStatus.APPROVED if best is not None and best.approved \
            else PlanStatus.NEEDS_REVISION
        return self._result(board, status)

    # ------------------------------------------------------------------ results
    def _result(self, board: CrewRunBoard, status: PlanStatus, *, error: str | None = None,
                question: str | None = None, rounds: int = 0, gave_up: bool = False
                ) -> PlanResult:
        req = board.requirements
        best = board.best if status in (PlanStatus.APPROVED, PlanStatus.NEEDS_REVISION,
                                        PlanStatus.FAILED, PlanStatus.CANCELLED) else None
        saved = self._persist(board, status) if status in (
            PlanStatus.APPROVED, PlanStatus.NEEDS_REVISION) else None
        result = PlanResult(
            session_id=board.session_id, run_id=board.run_id, mode=self.mode, status=status,
            request=req.request if req else None, draft=req.draft if req else None,
            itinerary=best.itinerary if best else None, review=best.review if best else None,
            violations=list(best.violations) if best else [], attempts=list(board.attempts),
            iterations=board.iteration, clarification_question=question,
            missing=list(req.missing) if req and status is PlanStatus.NEEDS_CLARIFICATION
            else [], assumptions=list(req.assumptions) if req else [],
            saved_version=saved, error=error)
        board.clarification_rounds, board.gave_up = rounds, gave_up  # stored by _finish
        return result

    def _persist(self, board: CrewRunBoard, status: PlanStatus) -> int | None:
        repo, best = self.repo, board.best
        if repo is None or best is None:
            return None
        record = repo.save_itinerary(best.itinerary, request=board.request,
                                     approved=status is PlanStatus.APPROVED,
                                     run_id=board.run_id)
        return record.version

    # ------------------------------------------------------------------ persistence
    def _record_start(self, session_id: str, run_id: str, message: str | None) -> None:
        if self.repo is None:
            return
        if self.repo.get_session(session_id) is None:
            title = (message or "Dubai trip").strip().splitlines()[0][:60] or "Dubai trip"
            self.repo.create_session(title=title, mode=self.mode, session_id=session_id)
        if message is not None:
            self.repo.add_message(session_id, ChatMessage(role=ChatRole.USER, content=message,
                                                          metadata={"run_id": run_id}))
        if self.repo.get_run(run_id) is None:
            self.repo.start_run(self.mode, session_id=session_id, run_id=run_id)

    def _finish(self, board: CrewRunBoard, result: PlanResult, stats: RunStats) -> None:
        if self.repo is None:
            return
        prior = self.repo.get_run(board.run_id)
        self.repo.finish_run(board.run_id, RUN_STATUS[result.status],
                             llm_calls=stats.llm_calls + (prior.llm_calls if prior else 0),
                             tokens=stats.tokens + (prior.tokens if prior else 0),
                             error=result.error)
        draft = result.draft.model_dump(mode="json") if result.draft else None
        meta: dict[str, Any] = {"run_id": board.run_id, "status": result.status.value,
                                "mode": self.mode.value, "draft": draft}
        if result.status is PlanStatus.NEEDS_CLARIFICATION:
            meta |= {"kind": "clarification", "missing": result.missing,
                     "clarifications": board.clarification_rounds, "gave_up": board.gave_up}
            text = result.clarification_question or DEFAULT_QUESTION
        elif result.status in (PlanStatus.APPROVED, PlanStatus.NEEDS_REVISION):
            meta |= {"kind": "plan_result", "saved_version": result.saved_version}
            text = summary_text(result.status, board.best, None, result.iterations)
        else:
            return
        self.repo.add_message(board.session_id, ChatMessage(role=ChatRole.ASSISTANT,
                                                            content=text, metadata=meta))


#: Agent-core name per role key (for ``LLM_MODEL__<AGENT>`` overrides).
STAGES_BY_ROLE: dict[str, str] = {s.role_key: s.core_agent or s.role_key
                                  for s in STAGES.values()}


# --------------------------------------------------------------------------- function API


@overload
def run_crewai(request_text: str | None, session_id: str, *, process: CrewProcessName = ...,
               stream: Literal[True] = ..., planner: CrewAIPlanner | None = ...,
               today: date | None = ...) -> Iterator[PlannerUpdate]: ...
@overload
def run_crewai(request_text: str | None, session_id: str, *, process: CrewProcessName = ...,
               stream: Literal[False], planner: CrewAIPlanner | None = ...,
               today: date | None = ...) -> PlanResult: ...
def run_crewai(request_text: str | None, session_id: str, *,
               process: CrewProcessName = "sequential", stream: bool = True,
               planner: CrewAIPlanner | None = None, today: date | None = None
               ) -> Iterator[PlannerUpdate] | PlanResult:
    """Plan ``request_text`` for ``session_id`` with CrewAI (same contract as LangGraph).

    ``stream=True`` yields :class:`PlannerUpdate`s (the last one carries the result);
    ``stream=False`` returns the :class:`PlanResult`. ``planner`` overrides ``process``.
    """
    service = planner or CrewAIPlanner(process=process)
    if not stream:
        return service.run(request_text, session_id, today=today)
    return service.stream(request_text, session_id, today=today)
