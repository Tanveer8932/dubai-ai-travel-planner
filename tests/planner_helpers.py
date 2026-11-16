"""Test helpers shared by the orchestration-mode tests (LangGraph and CrewAI)."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from typing import Any

from dubai_planner.agents import composer
from dubai_planner.models import Itinerary
from dubai_planner.tools.toolbox import Toolbox

TODAY = date(2026, 11, 16)
WINTER = "2 adults, 10-12 Feb 2027, AED 6000, history and food"
SUMMER = "2 adults, 14-16 July 2027, AED 9000, beach and culture"


class ComposerSpy:
    """Real composer; the first ``inflate`` attempts (None = all) come back over budget."""

    def __init__(self, inflate: int | None = 0) -> None:
        self.inflate = inflate
        self.calls: list[composer.ComposerInput] = []
        self.outputs: list[Itinerary] = []

    def __call__(self, data: composer.ComposerInput, llm: Any, tools: Toolbox) -> Itinerary:
        self.calls.append(data)
        itinerary = composer.run(data, llm, tools)
        if self.inflate is None or len(self.calls) <= self.inflate:
            total = data.request.total_budget_aed() * 2
            itinerary = itinerary.model_copy(update={
                "cost": itinerary.cost.model_copy(update={"total": total})})
        self.outputs.append(itinerary)
        return itinerary


class HeatDefectComposer:
    """First attempt: day 1 is one outdoor attraction at 12:00-13:30 (an LLM-style defect).

    The defective draft goes through the real composer's assembly (grounding, pricing),
    exactly like a model reply would; later attempts are the real composer.
    """

    def __init__(self) -> None:
        self.calls: list[composer.ComposerInput] = []
        self.outputs: list[Itinerary] = []

    def __call__(self, data: composer.ComposerInput, llm: Any, tools: Toolbox) -> Itinerary:
        self.calls.append(data)
        if len(self.calls) > 1:
            itinerary = composer.run(data, llm, tools)
        else:
            ctx = composer.build_context(data, tools)
            draft = composer.baseline_draft(ctx)
            outdoor = next(a for a in ctx.attractions if not a.indoor and not a.day_trip)
            midday = composer.DraftSlot(start="12:00", end="13:30", kind="attraction",
                                        ref_id=outdoor.id)
            first = draft.days[0].model_copy(update={"slots": [midday]})
            draft = draft.model_copy(update={"days": [first, *draft.days[1:]]})
            itinerary = composer.assemble(draft, data, tools)
        self.outputs.append(itinerary)
        return itinerary


def slow(output: Any, seconds: float = 1.0) -> Callable[..., Any]:
    """An agent stub that sleeps, then returns ``output`` (parallelism tests)."""

    def stub(*_args: Any, **_kwargs: Any) -> Any:
        time.sleep(seconds)
        return output

    return stub
