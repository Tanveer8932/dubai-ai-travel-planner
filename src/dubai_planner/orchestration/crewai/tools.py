"""CrewAI tools (implementation plan, Phase 4 task 2).

Two kinds, both thin wrappers:

- :class:`ToolboxTool` exposes one shared tool function from ``tools.toolbox.TOOL_SPECS``
  (fares, travel times, currency, weather…) with its Pydantic ``args_schema``.
- :class:`StageTool` is a specialist's main tool: it runs that role's shared agent
  (``agents.<x>.run``) on the inputs held by the run board, stores the typed output on
  the board and returns a short brief. ``result_as_answer=True`` makes that brief the
  agent's final answer, so the agent cannot rewrite tool output into invented venues
  or numbers, and the task's guardrail checks the typed output is really there.

A stage failure aborts the crew (``tool_failure_policy="raise"``). CrewAI re-tries a
raising tool a few times on its own, so the first failure is recorded on the board and
later attempts re-raise at once instead of repeating the agent's LLM work. Transient
provider errors (429, timeouts) are retried with bounded backoff *before* that.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

from crewai.tools import BaseTool
from crewai.tools.tool_failure import ToolFailurePolicy
from pydantic import BaseModel, Field, PrivateAttr

from dubai_planner.orchestration.common import RetrySettings, call_with_retry
from dubai_planner.orchestration.crewai.board import (
    STAGES,
    CrewRunAborted,
    CrewRunBoard,
    CrewRunCancelled,
    brief,
)
from dubai_planner.orchestration.crewai.stages import StageFn
from dubai_planner.tools.toolbox import TOOL_SPECS, Toolbox
from dubai_planner.tracing import agent_span, run_context


def _to_text(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, list):
        return json.dumps([v.model_dump(mode="json") if isinstance(v, BaseModel) else v
                           for v in value], default=str)
    return json.dumps(value, default=str)


def _strip_patterns(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_patterns(v) for k, v in node.items() if k != "pattern"}
    if isinstance(node, list):
        return [_strip_patterns(v) for v in node]
    return node


@cache
def provider_safe(model: type[BaseModel]) -> type[BaseModel]:
    """``model`` with a JSON schema that strict function-calling APIs accept.

    Pydantic describes ``Decimal`` fields with a regex using lookarounds, which Groq
    rejects (HTTP 400 ``pattern_unsupported_feature``, found in the Phase 4 smoke run).
    The subclass only drops ``pattern`` from the *advertised* schema; validation still
    uses the original field types.
    """

    class Safe(model):  # type: ignore[valid-type,misc]
        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            schema: dict[str, Any] = _strip_patterns(super().model_json_schema(*args, **kwargs))
            return schema

    Safe.__name__ = Safe.__qualname__ = model.__name__
    return Safe


class ToolboxTool(BaseTool):
    """One ``TOOL_SPECS`` entry as a CrewAI tool (validated args, JSON result)."""

    _toolbox: Toolbox = PrivateAttr()

    def __init__(self, toolbox: Toolbox, name: str, **data: Any) -> None:
        spec = TOOL_SPECS[name]
        super().__init__(name=name, description=spec.description,
                         args_schema=provider_safe(spec.args_model), **data)
        self._toolbox = toolbox

    def _run(self, **kwargs: Any) -> str:
        return _to_text(self._toolbox.call(self.name, kwargs))


def toolbox_tools(toolbox: Toolbox, names: list[str] | None = None) -> list[BaseTool]:
    """CrewAI wrappers for the shared tool functions (all of them when ``names`` is None)."""
    return [ToolboxTool(toolbox, name)
            for name in (list(TOOL_SPECS) if names is None else names)]


class StageArgs(BaseModel):
    notes: str = Field(default="", max_length=500,
                       description="Optional note for the trace; inputs come from the team")


class StageTool(BaseTool):
    """A specialist's main tool: runs its shared agent on the board's inputs."""

    args_schema: type[BaseModel] = StageArgs
    result_as_answer: bool = True
    tool_failure_policy: ToolFailurePolicy | None = ToolFailurePolicy.RAISE
    _board: CrewRunBoard = PrivateAttr()
    _stage: str = PrivateAttr()
    _fn: StageFn = PrivateAttr()
    _retry: RetrySettings = PrivateAttr()
    _role: str = PrivateAttr()

    def __init__(self, board: CrewRunBoard, stage: str, fn: StageFn, *, role: str,
                 retry: RetrySettings | None = None, **data: Any) -> None:
        spec = STAGES[stage]
        super().__init__(
            name=spec.tool,
            description=(f"Runs the {role}'s {spec.key.replace('_', ' ')} step on the "
                         "team's shared trip data and returns its result. Takes no "
                         "required input."), **data)
        self._board, self._stage, self._fn = board, stage, fn
        self._retry = retry or RetrySettings()
        self._role = role

    def _run(self, notes: str = "", **_ignored: Any) -> str:
        board, spec = self._board, STAGES[self._stage]
        if board.cancelled:
            raise CrewRunCancelled("the planning run was cancelled")
        if board.error is not None:  # CrewAI re-tries a raising tool: do not redo the work
            raise CrewRunAborted(f"stage {board.failed_stage!r} already failed")
        # Context from the board, not the calling thread: CrewAI may run tools in
        # worker threads, and every event must carry this run's id and iteration.
        with run_context(board.run_id, mode=board.mode.value, iteration=board.iteration):
            try:
                with agent_span(spec.task, kind="stage", role=self._role, tool=spec.tool,
                                notes=notes[:200]) as span:
                    value = call_with_retry(lambda: self._fn(board), self._retry,
                                            agent=spec.task)
                    span.update(kind="stage", output=type(value).__name__)
            except Exception as exc:
                board.fail(spec.key, exc)
                raise
        board.put_pending(spec.key, value)
        return brief(spec.key, value)
