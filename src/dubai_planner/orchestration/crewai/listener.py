"""CrewAI events → our :class:`TraceEvent`s (implementation plan, Phase 4 task 5).

CrewAI publishes typed events on its own bus (``crewai.events``). One process-wide
:class:`CrewTraceListener` turns those of *registered* crews into trace events on our
shared bus, attributed to the right run and role:

========================  =====================================================
CrewAI event              TraceEvent
========================  =====================================================
Task started/completed/   ``start``/``end``/``error``, agent = task name
failed                    (``payload.kind="crew_task"``, executing role)
Tool usage finished/      ``tool_call``, agent = role key of the agent that used
error                     the tool (``payload.tool``)
Delegation tool used      ``delegation``, agent = manager, ``payload.coworker``
LLM call completed by a   ``llm_call`` with token usage (``payload.kind=
crew agent                "crew_agent"``): the orchestration overhead
========================  =====================================================

CrewAI runs these handlers on its own thread pool, so the run is looked up from the
event's agent/task id (registered per crew), not from the handler thread's context;
timestamps come from the events themselves. :meth:`CrewTraceListener.flush` waits for
pending handlers before a run is finalised. Agent-core LLM calls are traced by the
structured-output helper and carry no CrewAI agent, so they are not counted twice.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from crewai.events import BaseEventListener, crewai_event_bus
from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.llm_events import LLMCallCompletedEvent
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.events.types.tool_usage_events import ToolUsageErrorEvent, ToolUsageFinishedEvent

from dubai_planner.models import TraceEvent, TraceEventType
from dubai_planner.orchestration.crewai.board import CrewRunBoard
from dubai_planner.tracing import bus

DELEGATION_TOOLS = {"delegate_work_to_coworker", "delegate work to coworker"}
QUESTION_TOOLS = {"ask_question_to_coworker", "ask question to coworker"}


@dataclass(frozen=True)
class _AgentInfo:
    board: CrewRunBoard
    role_key: str
    role: str


@dataclass(frozen=True)
class _TaskInfo:
    board: CrewRunBoard
    name: str
    role_key: str


def _aware(ts: datetime | None) -> datetime:
    if ts is None:
        return datetime.now(UTC)
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


class CrewTraceListener(BaseEventListener):
    """Maps CrewAI events of registered crews onto the shared trace bus."""

    def __init__(self) -> None:
        self._agents: dict[str, _AgentInfo] = {}
        self._tasks: dict[str, _TaskInfo] = {}
        self._task_started: dict[str, datetime] = {}
        self._lock = threading.Lock()
        super().__init__()

    # ------------------------------------------------------------------ registry
    def register(self, board: CrewRunBoard, *, agents: dict[str, tuple[Any, str]],
                 tasks: dict[str, tuple[Any, str]]) -> None:
        """``agents``: role key → (Agent, role); ``tasks``: task name → (Task, role key)."""
        with self._lock:
            for role_key, (agent, role) in agents.items():
                self._agents[str(agent.id)] = _AgentInfo(board, role_key, role)
            for name, (task, role_key) in tasks.items():
                self._tasks[str(task.id)] = _TaskInfo(board, name, role_key)

    def unregister(self, board: CrewRunBoard) -> None:
        with self._lock:
            self._agents = {k: v for k, v in self._agents.items() if v.board is not board}
            self._tasks = {k: v for k, v in self._tasks.items() if v.board is not board}

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait until CrewAI has delivered every pending event to the handlers."""
        return bool(crewai_event_bus.flush(timeout=timeout))

    # ------------------------------------------------------------------ helpers
    def _publish(self, board: CrewRunBoard, agent: str, kind: TraceEventType, *,
                 started_at: datetime, ended_at: datetime | None = None, tokens_in: int = 0,
                 tokens_out: int = 0, payload: dict[str, Any]) -> None:
        if ended_at is not None and ended_at < started_at:
            ended_at = started_at
        bus.publish(TraceEvent(
            run_id=board.run_id, mode=board.mode.value, agent=agent, event=kind,
            started_at=started_at, ended_at=ended_at, tokens_in=tokens_in,
            tokens_out=tokens_out, iteration=board.iteration, payload=payload))

    def _agent(self, agent_id: str | None) -> _AgentInfo | None:
        with self._lock:
            return self._agents.get(str(agent_id)) if agent_id else None

    def _task(self, task: Any) -> _TaskInfo | None:
        with self._lock:
            return self._tasks.get(str(getattr(task, "id", ""))) if task is not None else None

    # ------------------------------------------------------------------ handlers
    def on_task(self, event: TaskStartedEvent | TaskCompletedEvent | TaskFailedEvent) -> None:
        task = event.task
        info = self._task(task)
        if info is None or task is None:
            return
        task_id = str(task.id)
        executing = getattr(getattr(task, "agent", None), "role", None)
        payload: dict[str, Any] = {"kind": "crew_task", "role_key": info.role_key,
                                   "executed_by": executing}
        ts = _aware(event.timestamp)
        if isinstance(event, TaskStartedEvent):
            with self._lock:
                self._task_started[task_id] = ts
            self._publish(info.board, info.name, TraceEventType.START, started_at=ts,
                          payload=payload)
            return
        with self._lock:
            started = self._task_started.pop(task_id, ts)
        if isinstance(event, TaskFailedEvent):
            payload |= {"error": getattr(event.error_type, "__name__", "Exception")}
            self._publish(info.board, info.name, TraceEventType.ERROR, started_at=started,
                          ended_at=ts, payload=payload)
        else:
            self._publish(info.board, info.name, TraceEventType.END, started_at=started,
                          ended_at=ts, payload=payload | {"retries": task.retry_count})

    def on_tool(self, event: ToolUsageFinishedEvent | ToolUsageErrorEvent) -> None:
        info = self._agent(event.agent_id)
        if info is None:
            return
        name = event.tool_name.strip()
        args = event.tool_args if isinstance(event.tool_args, dict) else {}
        ok = isinstance(event, ToolUsageFinishedEvent) and event.failure is None
        if isinstance(event, ToolUsageFinishedEvent):
            started, ended = _aware(event.started_at), _aware(event.finished_at)
        else:
            started = ended = _aware(event.timestamp)
        if name.lower() in DELEGATION_TOOLS | QUESTION_TOOLS:
            self._publish(info.board, info.role_key, TraceEventType.DELEGATION,
                          started_at=started, ended_at=ended, payload={
                              "kind": "delegation", "role": info.role,
                              "coworker": str(args.get("coworker", ""))[:80],
                              "question": name.lower() in QUESTION_TOOLS,
                              "task": str(args.get("task") or args.get("question") or "")[:200],
                              "ok": ok})
            return
        self._publish(info.board, info.role_key, TraceEventType.TOOL_CALL, started_at=started,
                      ended_at=ended, payload={"kind": "tool", "tool": name, "role": info.role,
                                               "ok": ok, "task": (event.task_name or "")[:120]})

    def on_llm(self, event: LLMCallCompletedEvent) -> None:
        info = self._agent(event.agent_id)
        if info is None:  # agent-core calls carry no CrewAI agent; they are traced already
            return
        usage = event.usage or {}
        ts = _aware(event.timestamp)
        self._publish(info.board, info.role_key, TraceEventType.LLM_CALL, started_at=ts,
                      ended_at=ts, tokens_in=int(usage.get("prompt_tokens") or 0),
                      tokens_out=int(usage.get("completion_tokens") or 0),
                      payload={"kind": "crew_agent", "role": info.role,
                               "model": event.model or ""})

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        @crewai_event_bus.on(TaskStartedEvent)
        def _started(_source: Any, event: TaskStartedEvent) -> None:
            self.on_task(event)

        @crewai_event_bus.on(TaskCompletedEvent)
        def _completed(_source: Any, event: TaskCompletedEvent) -> None:
            self.on_task(event)

        @crewai_event_bus.on(TaskFailedEvent)
        def _failed(_source: Any, event: TaskFailedEvent) -> None:
            self.on_task(event)

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def _tool_done(_source: Any, event: ToolUsageFinishedEvent) -> None:
            self.on_tool(event)

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def _tool_error(_source: Any, event: ToolUsageErrorEvent) -> None:
            self.on_tool(event)

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _llm_done(_source: Any, event: LLMCallCompletedEvent) -> None:
            self.on_llm(event)


_listener: CrewTraceListener | None = None
_listener_lock = threading.Lock()


def trace_listener() -> CrewTraceListener:
    """The process-wide listener (handlers are registered on CrewAI's bus once)."""
    global _listener
    with _listener_lock:
        if _listener is None:
            _listener = CrewTraceListener()
        return _listener
