"""Mode-neutral planning service contract (architecture §2 ``PlannerService``).

The UI (Phase 5) and the evaluation runner (Phase 7) talk to a :class:`Planner` and
never branch on the framework. Both :class:`LangGraphPlanner` and
:class:`CrewAIPlanner` satisfy it and return the same :class:`PlanResult` /
:class:`PlannerUpdate` models; ``tests/integration/test_planner_contract.py`` checks
every mode against the same expectations (AC4.1).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dubai_planner.models import PlannerMode, PlannerUpdate, PlanResult


@runtime_checkable
class Planner(Protocol):
    """What every orchestration mode offers."""

    def stream(self, message: str | None, session_id: str, *, today: date | None = None
               ) -> Iterator[PlannerUpdate]:
        """Yield progress updates; the last one carries the :class:`PlanResult`."""
        ...

    def run(self, message: str | None, session_id: str, *, today: date | None = None
            ) -> PlanResult: ...

    def resume(self, session_id: str) -> PlanResult: ...

    def pending_question(self, session_id: str) -> str | None: ...


@contextmanager
def open_planner(mode: PlannerMode | str, *, checkpoint_path: Path | str | None = None,
                 **kwargs: Any) -> Iterator[Planner]:
    """A planner for ``mode`` (``langgraph``, ``crewai_sequential``, ``crewai_hierarchical``).

    ``kwargs`` (``llm``, ``tools``, ``repo``, ``agents``, ``settings``,
    ``max_review_iterations``…) go to the mode's planner. Frameworks are imported only
    for the mode that is used.
    """
    selected = PlannerMode(mode)
    if selected is PlannerMode.LANGGRAPH:
        from dubai_planner.orchestration.langgraph import LangGraphPlanner

        with LangGraphPlanner.open(checkpoint_path, **kwargs) as planner:
            yield planner
        return
    from dubai_planner.orchestration.crewai import CrewAIPlanner

    hierarchical = selected is PlannerMode.CREWAI_HIERARCHICAL
    yield CrewAIPlanner(process="hierarchical" if hierarchical else "sequential", **kwargs)
