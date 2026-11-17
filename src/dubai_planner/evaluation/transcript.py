"""Reproducible session transcript: a conversation, an app restart, and a persistence check.

:func:`record_session` plays a conversation through the app's own services
(``SessionService`` + ``RunManager`` → ``services.refinement.classify`` → planner),
recording for every turn the input, how it was routed, the steps that ran, the
outcome and how the itinerary changed (``SessionService.diff``). It then **closes**
the planners and the services, like quitting the app.

:func:`snapshot` is what a fresh process sees when the session is reopened: it reads
SQLite only (chat history, itinerary versions with parent links, the latest plan, the
runs). ``scripts/session_transcript.py`` calls it in a *separate* Python process and
:func:`verify` compares it with what was recorded before the restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import Field

from dubai_planner.models import ActivityKind, ItineraryRecord, PlannerMode
from dubai_planner.models.common import StrictModel
from dubai_planner.services.runs import PlannerHub, RunManager
from dubai_planner.services.session import SessionService
from dubai_planner.storage import Repository
from dubai_planner.tools.toolbox import Toolbox

TODAY = date(2026, 11, 16)
CONVERSATION = [
    ("Original request", "2 adults, 10-12 Feb 2027, AED 6000, history and food"),
    ("Refinement 1: budget", "reduce the budget by 20%"),
    ("Refinement 2: day 2 indoors", "Replace day 2 with indoor activities"),
    ("Refinement 3: dietary preference",
     "We are vegetarian now, please only choose vegetarian-friendly restaurants"),
]


class DaySummary(StrictModel):
    day: int
    date: str
    theme: str
    slots: list[str]
    day_cost_aed: str


class TurnLog(StrictModel):
    label: str
    user_input: str
    routing: str
    routing_explanation: str
    planner_message: str
    locked_days: list[int] = Field(default_factory=list)
    day_constraints: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    status: str | None = None
    iterations: int = 0
    llm_calls: int | None = None
    tokens: int | None = None
    duration_s: float | None = None
    saved_version: int | None = None
    parent_version: int | None = None
    budget_aed: str | None = None
    total_aed: str | None = None
    dietary: list[str] = Field(default_factory=list)
    diff: dict[str, Any] | None = None
    days: list[DaySummary] = Field(default_factory=list)
    vegetarian_restaurants_only: bool | None = None
    error: str | None = None


class Snapshot(StrictModel):
    """What a reopened app sees for the session (from SQLite only)."""

    session_id: str
    title: str
    messages: list[dict[str, str]]
    versions: list[dict[str, Any]]
    latest_version: int | None
    latest_total_aed: str | None
    latest_itinerary_json: str | None
    runs: list[dict[str, Any]]
    checkpoint_thread: bool | None = None


@dataclass
class Transcript:
    mode: str
    provider: str
    session_id: str
    db_path: str
    turns: list[TurnLog] = field(default_factory=list)
    before_close: Snapshot | None = None


def _days(record: ItineraryRecord) -> list[DaySummary]:
    return [DaySummary(day=i + 1, date=d.date.isoformat(), theme=d.theme,
                       slots=[f"{s.start:%H:%M}-{s.end:%H:%M} {s.activity.kind.value}: "
                              f"{s.activity.name}{'' if s.indoor else ' (outdoor)'}"
                              for s in d.slots], day_cost_aed=str(d.day_cost_aed))
            for i, d in enumerate(record.itinerary.days)]


def _vegetarian_only(record: ItineraryRecord, tools: Toolbox) -> bool:
    from dubai_planner.models import DietaryFlag, DiningOption

    venues = [tools.catalog.venue(s.activity.ref_id or "") for d in record.itinerary.days
              for s in d.slots if s.activity.kind is ActivityKind.DINING]
    return bool(venues) and all(isinstance(v, DiningOption)
                                and v.caters_for([DietaryFlag.VEGETARIAN]) for v in venues)


def snapshot(repo: Repository, session_id: str, *, checkpoint_path: Path | None = None
             ) -> Snapshot:
    session = repo.get_session(session_id)
    if session is None:
        raise LookupError(f"session {session_id} not found")
    versions = repo.list_itineraries(session_id)
    by_id = {v.id: v.version for v in versions}
    latest = versions[-1] if versions else None
    thread: bool | None = None
    if checkpoint_path is not None and checkpoint_path.exists():
        import sqlite3

        with sqlite3.connect(checkpoint_path) as conn:
            thread = conn.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                                  (session_id,)).fetchone()[0] > 0
    return Snapshot(
        session_id=session_id, title=session.title,
        messages=[{"role": m.role, "content": m.content}
                  for m in repo.list_messages(session_id)],
        versions=[{"version": v.version, "parent_version": by_id.get(v.parent_id)
                   if v.parent_id else None, "mode": v.mode.value, "approved": v.approved,
                   "total_aed": str(v.total_aed),
                   "budget_aed": str(v.request.total_budget_aed()) if v.request else None}
                  for v in versions],
        latest_version=latest.version if latest else None,
        latest_total_aed=str(latest.total_aed) if latest else None,
        latest_itinerary_json=latest.itinerary.model_dump_json() if latest else None,
        runs=[{"id": r.id, "status": r.status.value, "llm_calls": r.llm_calls,
               "tokens": r.tokens} for r in repo.list_runs(session_id)],
        checkpoint_thread=thread)


def record_session(db_path: Path, checkpoint_path: Path, mode: PlannerMode, *,
                   llm: Any = None, tools: Toolbox | None = None,
                   conversation: list[tuple[str, str]] | None = None,
                   turn_timeout_s: float = 1800) -> Transcript:
    """Play the conversation in a new session, then close everything (an app quit)."""
    tools = tools or Toolbox.default()
    repo = Repository(db_path)
    sessions = SessionService(repo)
    kwargs: dict[str, Any] = {"tools": tools}
    if llm is not None:
        kwargs["llm"] = llm
    hub = PlannerHub(repo, checkpoint_path=checkpoint_path, **kwargs)
    runs = RunManager(hub, sessions, today=lambda: TODAY)
    session = sessions.create(mode)
    log = Transcript(mode=mode.value, provider=hub.settings.llm_provider.value,
                     session_id=session.id, db_path=str(db_path))
    try:
        previous: ItineraryRecord | None = None
        for label, text in conversation or CONVERSATION:
            handle = runs.start(session.id, mode, text)
            finished = handle.wait(turn_timeout_s)
            ref, result = handle.refinement, handle.result
            edit = ref.edit
            turn = TurnLog(label=label, user_input=text, routing=ref.kind.value,
                           routing_explanation=ref.explanation, planner_message=ref.message,
                           locked_days=[d + 1 for d in (edit.locked_days if edit else [])],
                           day_constraints=[c.model_dump(mode="json")
                                            for c in (edit.day_constraints if edit else [])],
                           steps=[f"{u.node}" + (f" (iteration {u.iteration})"
                                                 if u.iteration else "")
                                  for u in handle.snapshot() if not u.is_final],
                           error=handle.error or (None if finished else "timed out"))
            if result is not None:
                run = repo.get_run(result.run_id)
                turn.status, turn.iterations = result.status.value, result.iterations
                turn.saved_version = result.saved_version
                if run is not None:
                    turn.llm_calls, turn.tokens = run.llm_calls, run.tokens
                    turn.duration_s = (round(run.duration_s, 2)
                                       if run.duration_s is not None else None)
                if result.error:
                    turn.error = result.error
            latest = sessions.latest(session.id)
            if latest is not None and (previous is None or latest.version != previous.version):
                turn.parent_version = previous.version if previous else None
                turn.total_aed = str(latest.total_aed)
                turn.budget_aed = (str(latest.request.total_budget_aed())
                                   if latest.request else None)
                turn.dietary = [d.value for d in latest.request.dietary] if latest.request else []
                turn.days = _days(latest)
                turn.vegetarian_restaurants_only = _vegetarian_only(latest, tools)
                if previous is not None:
                    diff = sessions.diff(previous, latest)
                    turn.diff = {
                        "from_version": diff.old_version, "to_version": diff.new_version,
                        "cost_delta_aed": {k: str(v) for k, v in diff.cost_delta.items()},
                        "budget_before": str(diff.budget_before),
                        "budget_after": str(diff.budget_after),
                        "changed_days": [d + 1 for d in diff.changed_days],
                        "unchanged_days": [d + 1 for d in diff.unchanged_days],
                        "slots": {str(d.day_index + 1): {
                            "removed": [s.label() for s in d.removed],
                            "added": [s.label() for s in d.added]}
                            for d in diff.days if d.changed}}
                previous = latest
            log.turns.append(turn)
        log.before_close = snapshot(repo, session.id, checkpoint_path=checkpoint_path)
    finally:
        hub.close()  # quit: planners and checkpointer closed, nothing kept in memory
    return log


class Check(StrictModel):
    name: str
    passed: bool
    detail: str = ""


def verify(before: Snapshot, after: Snapshot) -> list[Check]:
    """Compare the session before the app was closed with what a new process reads."""
    def check(name: str, ok: bool, detail: str = "") -> Check:
        return Check(name=name, passed=ok, detail=detail)

    versions = [v["version"] for v in after.versions]
    parents = [v["parent_version"] for v in after.versions]
    return [
        check("chat history identical", before.messages == after.messages,
              f"{len(after.messages)} messages"),
        check("every version still listed", before.versions == after.versions,
              f"versions {versions}"),
        check("versions are linked parent → child",
              parents == [None, *versions[:-1]], f"parents {parents}"),
        check("latest itinerary identical",
              before.latest_itinerary_json == after.latest_itinerary_json
              and after.latest_itinerary_json is not None,
              f"v{after.latest_version}, AED {after.latest_total_aed}"),
        check("run records kept", before.runs == after.runs, f"{len(after.runs)} runs"),
        *([check("LangGraph checkpoint thread kept", bool(after.checkpoint_thread))]
          if before.checkpoint_thread is not None else []),
    ]
