"""CrewAI task guardrails (implementation plan, Phase 4 task 4).

Every task has one. A guardrail is CrewAI's native "validate, else send feedback and
retry the task" hook (``Task(guardrail=..., guardrail_max_retries=...)``):

- :class:`StageGuardrail` (every task): accepts only the typed output the role's tool
  put on the board during *this* attempt and attaches it as ``TaskOutput.pydantic``.
  An agent that skipped its tool and wrote an answer itself is rejected with feedback.
  A recorded stage failure or a cancellation stops the crew.
- :class:`ItineraryGuardrail` (compose task): the deterministic validators. An
  itinerary with an error-severity rule violation is rejected with targeted feedback
  (the shared ``plan_revision``), CrewAI re-runs the task and the Itinerary Architect
  calls the composer again with that feedback. Each rejection is one refinement and
  counts against ``MAX_REVIEW_ITERATIONS``; at the cap the attempt is accepted *as
  unapproved* so the loop always terminates.
- :class:`ReviewGuardrail` (review task): records the attempt. Approval means the
  reviewer approved **and** no error-severity rule is violated (``Attempt.approved``),
  so a reviewer can never approve a hard-rule violation.

CrewAI receives the bound ``check`` method (it inspects guardrails as functions, so a
callable object is rejected). This module deliberately has no
``from __future__ import annotations``: CrewAI checks a guardrail's return annotation
at runtime and needs the real ``tuple[bool, Any]``.
"""

from typing import Any

from crewai.tasks.task_output import TaskOutput

from dubai_planner.agents import reviewer
from dubai_planner.agents.revision import plan_revision
from dubai_planner.models import Attempt, Itinerary, ReviewReport, Severity, TraceEventType
from dubai_planner.orchestration.crewai.board import STAGES, CrewRunBoard, brief
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import emit, run_context
from dubai_planner.validation.rules import validate_itinerary


class StageGuardrail:
    """Accept only the output the stage's tool produced in this attempt."""

    def __init__(self, board: CrewRunBoard, stage: str) -> None:
        self.board = board
        self.stage = STAGES[stage]
        self.checks = 0

    def _event(self, passed: bool, **payload: Any) -> None:
        with run_context(self.board.run_id, mode=self.board.mode.value,
                         iteration=self.board.iteration):
            emit(self.stage.task, TraceEventType.GUARDRAIL, payload={
                "kind": "guardrail", "passed": passed, "check": self.checks,
                "role_key": self.stage.role_key, **payload})

    def _missing(self) -> tuple[bool, Any]:
        self._event(False, reason="no_tool_result")
        return (False, f"Your answer was not produced by the `{self.stage.tool}` tool. Call "
                       f"the `{self.stage.tool}` tool and give its result as your final "
                       "answer, unchanged. Do not write the result yourself.")

    def _accept(self, output: TaskOutput, value: Any) -> tuple[bool, Any]:
        self.board.accept(self.stage.key, value)
        return (True, output.model_copy(update={"pydantic": value,
                                                "raw": brief(self.stage.key, value)}))

    def check(self, output: TaskOutput) -> tuple[bool, Any]:
        self.checks += 1
        self.board.raise_if_stopped()
        value = self.board.take_pending(self.stage.key)
        if value is None:
            return self._missing()
        self._event(True)
        return self._accept(output, value)


class ItineraryGuardrail(StageGuardrail):
    """Deterministic validators on the composed itinerary; reject hard-rule errors."""

    def __init__(self, board: CrewRunBoard, tools: Toolbox) -> None:
        super().__init__(board, "compose")
        self.tools = tools

    def check(self, output: TaskOutput) -> tuple[bool, Any]:
        self.checks += 1
        board = self.board
        board.raise_if_stopped()
        itinerary = board.take_pending("compose")
        if not isinstance(itinerary, Itinerary):
            return self._missing()
        violations = validate_itinerary(itinerary, board.request, board.research(),
                                        catalog=self.tools.catalog)
        errors = [v for v in violations if v.severity is Severity.ERROR]
        board.itinerary, board.violations = itinerary, violations
        rules = sorted({v.rule_id for v in errors})
        if errors and board.can_revise:
            # Rule-only review from the shared reviewer (no LLM call: the reviewer
            # could not approve a hard-rule error anyway).
            report = reviewer.merge(reviewer.deterministic_draft(violations), violations,
                                    board.iteration)
            board.record_attempt(Attempt(iteration=board.iteration, itinerary=itinerary,
                                         review=report, violations=violations))
            plan = plan_revision(report, itinerary, num_days=board.request.num_days)
            self._event(False, reason="rule_errors", rules=rules, retry=board.iteration + 1,
                        target_days=plan.target_days, locked_days=plan.locked_days)
            board.start_revision(plan)
            board.guardrail_rejections += 1
            return (False, "Rejected by the deterministic checks:\n- "
                    + "\n- ".join(plan.feedback)
                    + f"\nCall the `{self.stage.tool}` tool again; it applies this feedback "
                      "and keeps the days that passed.")
        self._event(True, rules=rules, cap_reached=bool(errors))
        return self._accept(output, itinerary)


class ReviewGuardrail(StageGuardrail):
    """Record the reviewed attempt; approval can never override a hard-rule error."""

    def __init__(self, board: CrewRunBoard) -> None:
        super().__init__(board, "review")

    def check(self, output: TaskOutput) -> tuple[bool, Any]:
        self.checks += 1
        board = self.board
        board.raise_if_stopped()
        report = board.take_pending("review")
        if not isinstance(report, ReviewReport):
            return self._missing()
        if board.itinerary is None:  # pragma: no cover - the review tool checks this first
            raise RuntimeError("no itinerary was reviewed")
        attempt = Attempt(iteration=board.iteration, itinerary=board.itinerary,
                          review=report, violations=list(board.violations))
        board.record_attempt(attempt)
        self._event(True, approved=attempt.approved,
                    reviewer_approved=report.approved,
                    errors=attempt.errors)
        return self._accept(output, report)
