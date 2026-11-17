"""Session state for the UI (implementation plan, Phase 5 task 5).

Everything a page shows comes from SQLite through this service: sessions, the
conversation, run records and itinerary versions. ``st.session_state`` only holds the
current ``session_id`` and UI flags, so a restarted app shows the same sessions.
Ownership (who may see which session) is enforced here, not in the pages: see
:class:`SessionService` and :mod:`dubai_planner.services.ownership`.
"""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

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
from dubai_planner.services import observability as ob
from dubai_planner.services.ownership import Owner, SessionNotFound
from dubai_planner.services.presentation import export_json, export_markdown
from dubai_planner.storage import Repository

if TYPE_CHECKING:
    from dubai_planner.data.seed import SeedCatalog

ExportFormat = Literal["json", "markdown"]

DEFAULT_TITLE = "New trip"
MAX_TITLE = 80
COST_FIELDS = ("accommodation", "activities", "food", "transport", "contingency", "total")


def new_session_id() -> str:
    return uuid.uuid4().hex


def title_from(message: str) -> str:
    first = message.strip().splitlines()[0] if message.strip() else ""
    return (first[:57] + "…") if len(first) > 60 else (first or DEFAULT_TITLE)


class SessionService:
    """Sessions, conversations, itinerary versions and runs, read from SQLite.

    Every call is made on behalf of an :class:`Owner`. The process-wide instance is
    :attr:`Owner.LOCAL` (unrestricted: scripts, tests, ``SESSION_SCOPE=shared``); a page
    serving a visitor uses :meth:`scoped`. A scoped service enforces ownership on every
    read and write: a session or run of another owner behaves exactly like a missing
    one (``get`` -> ``None``, everything else -> :class:`SessionNotFound`), and rename /
    delete repeat the owner condition in SQL.
    """

    def __init__(self, repo: Repository, owner: Owner = Owner.LOCAL) -> None:
        self.repo = repo
        self.owner = owner

    def scoped(self, owner: Owner) -> SessionService:
        """The same database, seen by ``owner``."""
        return SessionService(self.repo, owner)

    # ------------------------------------------------------------------ access
    def owns(self, session_id: str) -> bool:
        """May this owner use ``session_id``? (Local: whenever it exists.)"""
        if self.owner.key is None:
            return self.repo.get_session(session_id) is not None
        return self.repo.owns_session(session_id, self.owner.key)

    def _require(self, session_id: str) -> None:
        if self.owner.key is not None and not self.repo.owns_session(session_id,
                                                                     self.owner.key):
            raise SessionNotFound

    def check_writable(self, session_id: str) -> None:
        """Raise unless ``session_id`` is this owner's or does not exist yet."""
        if (self.owner.key is not None and self.repo.get_session(session_id) is not None
                and not self.repo.owns_session(session_id, self.owner.key)):
            raise SessionNotFound

    # ------------------------------------------------------------------ sessions
    def list_sessions(self) -> list[Session]:
        return sorted(self.repo.list_sessions(owner=self.owner.key),
                      key=lambda s: s.updated_at, reverse=True)

    def get(self, session_id: str) -> Session | None:
        if self.owner.key is not None and not self.repo.owns_session(session_id,
                                                                     self.owner.key):
            return None
        return self.repo.get_session(session_id)

    def create(self, mode: PlannerMode, title: str = DEFAULT_TITLE) -> Session:
        return self.repo.create_session(title=title.strip()[:MAX_TITLE] or DEFAULT_TITLE,
                                        mode=mode, session_id=new_session_id(),
                                        owner=self.owner.key)

    def ensure(self, session_id: str, mode: PlannerMode) -> Session:
        self.check_writable(session_id)
        return self.get(session_id) or self.repo.create_session(
            title=DEFAULT_TITLE, mode=mode, session_id=session_id, owner=self.owner.key)

    def rename(self, session_id: str, title: str) -> bool:
        clean = title.strip()[:MAX_TITLE]
        if not clean:
            raise ValueError("the title must not be empty")
        self._require(session_id)
        return self.repo.rename_session(session_id, clean, owner=self.owner.key)

    def title_from_first_message(self, session_id: str, message: str) -> None:
        """Give an untitled session the first request as its title."""
        session = self.get(session_id)
        if session is not None and session.title == DEFAULT_TITLE:
            self.repo.rename_session(session_id, title_from(message), owner=self.owner.key)

    def delete(self, session_id: str) -> bool:
        """Delete the session with its messages, runs and versions (SQLite cascade)."""
        self._require(session_id)
        return self.repo.delete_session(session_id, owner=self.owner.key)

    # ------------------------------------------------------------------ content
    def history(self, session_id: str) -> list[ChatMessage]:
        self._require(session_id)
        return self.repo.chat_history(session_id)

    def runs(self, session_id: str) -> list[RunRecord]:
        self._require(session_id)
        return self.repo.list_runs(session_id)

    def versions(self, session_id: str) -> list[ItineraryRecord]:
        self._require(session_id)
        return self.repo.list_itineraries(session_id)

    def latest(self, session_id: str) -> ItineraryRecord | None:
        self._require(session_id)
        return self.repo.latest_itinerary(session_id)

    def version(self, session_id: str, number: int) -> ItineraryRecord | None:
        return next((r for r in self.versions(session_id) if r.version == number), None)

    def compare_versions(self, session_id: str, old: int, new: int) -> VersionDiff:
        """Diff two versions of one of this owner's sessions."""
        a, b = self.version(session_id, old), self.version(session_id, new)
        if a is None or b is None:
            raise LookupError("version not found")
        return self.diff(a, b)

    def export_version(self, session_id: str, number: int, fmt: ExportFormat,
                       catalog: SeedCatalog) -> str:
        """A saved version as JSON or Markdown (request and itinerary only)."""
        record = self.version(session_id, number)
        if record is None:
            raise LookupError("version not found")
        return export_json(record) if fmt == "json" else export_markdown(record, catalog)

    # ------------------------------------------------------------------ runs and traces
    def run_records(self, session_id: str | None = None) -> list[RunRecord]:
        """Runs newest first: one session, or every run this owner may see."""
        if session_id is not None:
            runs = self.runs(session_id)
        elif self.owner.key is None:
            runs = self.repo.list_runs()
        else:
            runs = self.repo.list_owner_runs(self.owner.key)
        return sorted(runs, key=lambda r: r.started_at, reverse=True)

    def get_run(self, run_id: str) -> RunRecord | None:
        """The run, if it belongs to one of this owner's sessions (local: any run)."""
        run = self.repo.get_run(run_id)
        if run is None or self.owner.key is None:
            return run
        if run.session_id is None or not self.repo.owns_session(run.session_id,
                                                                self.owner.key):
            return None
        return run

    def trace(self, run_id: str, *, active: bool | None = None) -> ob.RunTrace | None:
        """A run's record and trace events, if this owner may see the run."""
        if self.get_run(run_id) is None:
            return None
        return ob.load_trace(self.repo, run_id, active=active)

    def export_trace(self, run_id: str) -> str:
        """A run's record and events as redacted JSON."""
        trace = self.trace(run_id, active=False)
        if trace is None:
            raise SessionNotFound
        return ob.export_trace(trace)

    def run_messages(self, runs: list[RunRecord]) -> dict[str, str]:
        """The user message that started each of this owner's ``runs``."""
        visible = [r for r in runs if self.get_run(r.id) is not None]
        return ob.messages_by_run(self.repo, visible)

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
