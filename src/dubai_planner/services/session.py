"""Session state for the UI (implementation plan, Phase 5 task 5).

Everything a page shows comes from SQLite through this service: sessions, the
conversation, run records and itinerary versions. ``st.session_state`` only holds the
current ``session_id`` and UI flags, so a restarted app shows the same sessions.
"""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

from dubai_planner.models import (
    ChatMessage,
    DayDiff,
    DayPlan,
    Itinerary,
    ItineraryRecord,
    PlannerMode,
    RunRecord,
    Session,
    SlotRef,
    TimeSlot,
    VersionDiff,
)
from dubai_planner.storage import Repository

DEFAULT_TITLE = "New trip"
MAX_TITLE = 80
COST_FIELDS = ("accommodation", "activities", "food", "transport", "contingency", "total")


def new_session_id() -> str:
    return uuid.uuid4().hex


def title_from(message: str) -> str:
    first = message.strip().splitlines()[0] if message.strip() else ""
    return (first[:57] + "…") if len(first) > 60 else (first or DEFAULT_TITLE)


class SessionService:
    """Sessions, conversations and itinerary versions, read from SQLite."""

    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    # ------------------------------------------------------------------ sessions
    def list_sessions(self) -> list[Session]:
        return sorted(self.repo.list_sessions(), key=lambda s: s.updated_at, reverse=True)

    def get(self, session_id: str) -> Session | None:
        return self.repo.get_session(session_id)

    def create(self, mode: PlannerMode, title: str = DEFAULT_TITLE) -> Session:
        return self.repo.create_session(title=title.strip()[:MAX_TITLE] or DEFAULT_TITLE,
                                        mode=mode, session_id=new_session_id())

    def ensure(self, session_id: str, mode: PlannerMode) -> Session:
        return self.get(session_id) or self.repo.create_session(
            title=DEFAULT_TITLE, mode=mode, session_id=session_id)

    def rename(self, session_id: str, title: str) -> bool:
        clean = title.strip()[:MAX_TITLE]
        if not clean:
            raise ValueError("the title must not be empty")
        return self.repo.rename_session(session_id, clean)

    def title_from_first_message(self, session_id: str, message: str) -> None:
        """Give an untitled session the first request as its title."""
        session = self.get(session_id)
        if session is not None and session.title == DEFAULT_TITLE:
            self.repo.rename_session(session_id, title_from(message))

    def delete(self, session_id: str) -> bool:
        """Delete the session with its messages, runs and versions (SQLite cascade)."""
        return self.repo.delete_session(session_id)

    # ------------------------------------------------------------------ content
    def history(self, session_id: str) -> list[ChatMessage]:
        return self.repo.chat_history(session_id)

    def runs(self, session_id: str) -> list[RunRecord]:
        return self.repo.list_runs(session_id)

    def versions(self, session_id: str) -> list[ItineraryRecord]:
        return self.repo.list_itineraries(session_id)

    def latest(self, session_id: str) -> ItineraryRecord | None:
        return self.repo.latest_itinerary(session_id)

    def version(self, session_id: str, number: int) -> ItineraryRecord | None:
        return next((r for r in self.versions(session_id) if r.version == number), None)

    # ------------------------------------------------------------------ diff
    @staticmethod
    def diff(old: ItineraryRecord, new: ItineraryRecord) -> VersionDiff:
        """What changed from ``old`` to ``new``: cost per category and slots per day."""
        a, b = old.itinerary, new.itinerary
        delta = {f: Decimal(getattr(b.cost, f)) - Decimal(getattr(a.cost, f))
                 for f in COST_FIELDS}
        days = []
        for i in range(max(len(a.days), len(b.days))):
            before = a.days[i] if i < len(a.days) else None
            after = b.days[i] if i < len(b.days) else None
            old_slots = [_ref(s) for s in before.slots] if before else []
            new_slots = [_ref(s) for s in after.slots] if after else []
            days.append(DayDiff(
                day_index=i, date=_day_date(before, after),
                added=[s for s in new_slots if s not in old_slots],
                removed=[s for s in old_slots if s not in new_slots],
                theme_before=before.theme if before else "",
                theme_after=after.theme if after else "",
                cost_before=before.day_cost_aed if before else Decimal("0"),
                cost_after=after.day_cost_aed if after else Decimal("0")))
        return VersionDiff(session_id=new.session_id, old_version=old.version,
                           new_version=new.version, cost_delta=delta,
                           budget_before=_budget(old), budget_after=_budget(new), days=days)


def _day_date(before: DayPlan | None, after: DayPlan | None) -> date | None:
    day = after or before
    return day.date if day is not None else None


def _ref(slot: TimeSlot) -> SlotRef:
    return SlotRef(start=slot.start, end=slot.end, kind=slot.activity.kind.value,
                   name=slot.activity.name, ref_id=slot.activity.ref_id, cost_aed=slot.cost_aed)


def _budget(record: ItineraryRecord) -> Decimal | None:
    if record.request is not None:
        return record.request.total_budget_aed()
    return _itinerary_budget(record.itinerary)


def _itinerary_budget(itinerary: Itinerary) -> Decimal | None:
    return itinerary.cost.budget_aed
