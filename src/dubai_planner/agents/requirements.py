"""Requirements agent: natural-language request (+ history) → ``TripRequestDraft``.

The LLM fills :class:`RequirementsExtraction`, a flat schema close to how people talk
(``duration_days``, ``month``, ``budget_amount`` + ``budget_currency``). Python then
resolves it deterministically:

- dates: explicit dates > start + duration > month (+ year) + duration (1st of the next
  such month, recorded as an assumption); past dates are dropped and asked for again;
- budget: converted to AED with the currency tool (USD via the 3.6725 peg, offline);
- vocabulary: dietary/accessibility/pace words normalised to the enums;
- clarification: merged with the previous draft (new values win, lists are unioned).

The output says which required fields are still missing and asks a question for them
(the LangGraph clarification gate in Phase 3 uses this).
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from pydantic import Field, field_validator

from dubai_planner.agents.base import LLMOutput, context_block, load_prompt, user_text_block
from dubai_planner.llm.mock import MockRequest, register_generator
from dubai_planner.llm.structured import build_messages, invoke_structured
from dubai_planner.models.common import StrictModel
from dubai_planner.models.planning import PlanEdit
from dubai_planner.models.request import (
    MAX_TRIP_DAYS,
    AccessibilityNeed,
    BudgetScope,
    DietaryFlag,
    Pace,
    Travellers,
    TripRequest,
    TripRequestDraft,
)
from dubai_planner.models.trace import ChatMessage, ChatRole
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import agent_span

AGENT = "requirements"
HISTORY_TURNS = 6
DEFAULT_CHILD_AGE = 8

_ALIASES: dict[str, str] = {
    "veg": "vegetarian", "veggie": "vegetarian", "gluten": "gluten_free",
    "gluten free": "gluten_free", "glutenfree": "gluten_free", "coeliac": "gluten_free",
    "celiac": "gluten_free", "wheelchair user": "wheelchair",
    "reduced mobility": "limited_mobility",
    "mobility": "limited_mobility", "pram": "stroller", "buggy": "stroller", "slow": "relaxed",
    "easy": "relaxed", "chill": "relaxed", "busy": "packed", "intense": "packed",
    "medium": "moderate", "normal": "moderate", "balanced": "moderate",
}


def _norm(value: str) -> str:
    key = value.strip().lower().replace("-", " ").replace("_", " ")
    return _ALIASES.get(key, key).replace(" ", "_")


def _enum_list(values: Any, enum: type[DietaryFlag] | type[AccessibilityNeed]) -> list[str]:
    allowed = {e.value for e in enum}
    items = values if isinstance(values, list) else [values] if values else []
    return list(dict.fromkeys(n for v in items if isinstance(v, str)
                              if (n := _norm(v)) in allowed))


class RequirementsExtraction(LLMOutput):
    """What the LLM extracts. Leave a field null/empty when the user did not say it."""

    start_date: date | None = None
    end_date: date | None = None
    duration_days: int | None = Field(default=None, ge=1, le=MAX_TRIP_DAYS)
    month: int | None = Field(default=None, ge=1, le=12, description="If only a month is given")
    year: int | None = Field(default=None, ge=2024, le=2100)
    adults: int | None = Field(default=None, ge=1, le=20)
    children: int | None = Field(default=None, ge=0, le=20)
    children_ages: list[int] = Field(default_factory=list)
    budget_amount: Decimal | None = Field(default=None, gt=0)
    budget_currency: str = Field(default="AED", min_length=3, max_length=3)
    budget_scope: BudgetScope | None = None
    interests: list[str] = Field(default_factory=list)
    pace: Pace | None = None
    dietary: list[DietaryFlag] = Field(default_factory=list)
    accessibility: list[AccessibilityNeed] = Field(default_factory=list)
    preferred_areas: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=1000)

    @field_validator("dietary", mode="before")
    @classmethod
    def _dietary(cls, v: Any) -> list[str]:
        return _enum_list(v, DietaryFlag)

    @field_validator("accessibility", mode="before")
    @classmethod
    def _access(cls, v: Any) -> list[str]:
        return _enum_list(v, AccessibilityNeed)

    @field_validator("pace", "budget_scope", mode="before")
    @classmethod
    def _norm_enum(cls, v: Any) -> Any:
        return _norm(v) if isinstance(v, str) and v.strip() else (None if v == "" else v)

    @field_validator("budget_currency", mode="before")
    @classmethod
    def _ccy(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "AED"
        aliases = {"DHS": "AED", "DH": "AED", "DIRHAM": "AED", "DIRHAMS": "AED", "$": "USD",
                   "€": "EUR", "£": "GBP", "RS": "INR", "RUPEES": "INR"}
        text = str(v).strip().upper()
        return aliases.get(text, text)

    @field_validator("interests", "preferred_areas", "must_include", "avoid", mode="before")
    @classmethod
    def _str_list(cls, v: Any) -> list[str]:
        items = v if isinstance(v, list) else [v] if v else []
        return list(dict.fromkeys(str(i).strip() for i in items if str(i).strip()))[:15]

    @field_validator("children_ages", mode="before")
    @classmethod
    def _ages(cls, v: Any) -> list[int]:
        items = v if isinstance(v, list) else [v] if v not in (None, "") else []
        return [int(a) for a in items if str(a).strip().isdigit() and 0 <= int(a) <= 17]


class RequirementsInput(StrictModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)
    previous: TripRequestDraft | None = None
    today: date = Field(default_factory=date.today)


class RequirementsResult(StrictModel):
    draft: TripRequestDraft
    request: TripRequest | None = None
    missing: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    assumptions: list[str] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.request is not None


QUESTIONS = {
    "start_date": "When does your trip start (date or month)?",
    "end_date": "How many days will you stay, or when do you leave?",
    "budget_aed": "What is your budget (amount and currency, total or per person)?",
}


def clarification_for(missing: list[str]) -> str | None:
    if not missing:
        return None
    asks = list(dict.fromkeys(QUESTIONS[m] for m in missing if m in QUESTIONS))
    return "To plan your Dubai trip I still need: " + " ".join(asks)


def _next_month_start(month: int, year: int | None, today: date) -> date:
    candidate = date(year or today.year, month, 1)
    if year is None and candidate < today.replace(day=1):
        candidate = date(today.year + 1, month, 1)
    if candidate < today:  # this month: start tomorrow-ish rather than in the past
        candidate = today + timedelta(days=1)
    return candidate


def resolve(ex: RequirementsExtraction, data: RequirementsInput, tools: Toolbox
            ) -> RequirementsResult:
    """Deterministic post-processing (raises ``ValueError`` for contradictory output)."""
    assumptions: list[str] = []
    today = data.today
    start, end = ex.start_date, ex.end_date
    if start is None and ex.month is not None:
        start = _next_month_start(ex.month, ex.year, today)
        if ex.duration_days:
            assumptions.append(f"Only the month was given; assumed the trip starts "
                               f"{start.isoformat()}.")
    if start is not None and end is None and ex.duration_days:
        end = start + timedelta(days=ex.duration_days - 1)
    if start is not None and start < today:
        assumptions.append(f"Start date {start.isoformat()} is in the past and was ignored.")
        start, end = None, None
    if start is not None and end is not None and end < start:
        raise ValueError(f"end_date {end} is before start_date {start}; for a duration use "
                         "duration_days")

    budget_aed: Decimal | None = None
    if ex.budget_amount is not None:
        try:
            budget_aed = tools.convert_currency(ex.budget_amount, ex.budget_currency, "AED")
        except ValueError as exc:
            raise ValueError(f"budget_currency {ex.budget_currency!r} is not supported "
                             "(use ISO codes such as AED, USD, EUR)") from exc
        if ex.budget_currency != "AED":
            assumptions.append(f"Budget {ex.budget_amount} {ex.budget_currency} ≈ AED "
                               f"{budget_aed} (converted).")

    travellers: Travellers | None = None
    if ex.adults is not None or ex.children or ex.children_ages:
        ages = list(ex.children_ages)
        if ex.children and len(ages) < ex.children:
            assumptions.append(f"Children's ages not given; assumed {DEFAULT_CHILD_AGE}.")
            ages += [DEFAULT_CHILD_AGE] * (ex.children - len(ages))
        travellers = Travellers(adults=ex.adults or 1, children_ages=ages)

    new = TripRequestDraft(
        start_date=start, end_date=end, travellers=travellers, budget_aed=budget_aed,
        budget_scope=ex.budget_scope, interests=ex.interests, pace=ex.pace, dietary=ex.dietary,
        accessibility=ex.accessibility, preferred_areas=ex.preferred_areas,
        must_include=ex.must_include, avoid=ex.avoid, notes=ex.notes,
    )
    draft = merge_drafts(data.previous, new)
    missing = draft.missing_fields()
    request = draft.to_request() if not missing else None  # ValueError → repair
    return RequirementsResult(draft=draft, request=request, missing=missing,
                              clarification_question=clarification_for(missing),
                              assumptions=assumptions)


def merge_drafts(previous: TripRequestDraft | None, new: TripRequestDraft) -> TripRequestDraft:
    """New non-empty values win; list fields are unioned in order."""
    if previous is None:
        return new
    merged = previous.model_dump()
    for name, value in new.model_dump().items():
        if isinstance(value, list):
            merged[name] = list(dict.fromkeys([*merged.get(name, []), *value]))
        elif value not in (None, ""):
            merged[name] = value
    return TripRequestDraft.model_validate(merged)


def run(data: RequirementsInput, llm: Any, tools: Toolbox) -> RequirementsResult:
    with agent_span(AGENT) as span:
        recent = [m for m in data.history if m.role in (ChatRole.USER, ChatRole.ASSISTANT)]
        history = "\n".join(f"{m.role.value}: {m.content[:500]}" for m in recent[-HISTORY_TURNS:])
        ctx = {"today": data.today.isoformat(),
               "previous": data.previous.model_dump(mode="json") if data.previous else None}
        messages = build_messages(
            system=load_prompt("requirements.system", today=data.today.isoformat()),
            user=load_prompt("requirements.user", context=context_block(ctx),
                             history=history or "(none)",
                             request=user_text_block(data.message)),
            schema=RequirementsExtraction, agent=AGENT,
        )
        result = invoke_structured(llm, messages, RequirementsExtraction, agent=AGENT,
                                   postprocess=lambda ex: resolve(ex, data, tools)).value
        span.update(missing=result.missing, complete=result.is_complete)
        return result


def unchanged(edit: PlanEdit) -> RequirementsResult | None:
    """The edit's base request as a result, without an LLM call (local edits keep it)."""
    if not edit.keep_request or edit.base_request is None:
        return None
    draft = edit.base_draft()
    assert draft is not None  # noqa: S101 - base_request is set
    with agent_span(AGENT, reused=True) as span:
        span.update(missing=[], complete=True, reused=True)
        return RequirementsResult(draft=draft, request=edit.base_request)


def _mock_requirements(req: MockRequest) -> RequirementsExtraction:
    from dubai_planner.agents.requirements_heuristic import heuristic_extract

    today = date.fromisoformat(str((req.context or {}).get("today") or date.today()))
    return heuristic_extract(req.user_text, today=today)


register_generator(RequirementsExtraction.__name__, _mock_requirements)
