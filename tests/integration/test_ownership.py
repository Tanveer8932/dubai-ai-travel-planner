"""Deployment security: sessions and runs are isolated per visitor (service layer).

The pages only filter what they show; these tests prove the *services* refuse another
owner's data whatever id a caller passes (direct links, tampered UI state).
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from factories import itinerary, trip_request
from planner_helpers import TODAY, WINTER

from dubai_planner.config import SessionScope, Settings
from dubai_planner.llm.mock import MockLLM
from dubai_planner.models import ChatMessage, ChatRole, PlannerMode, TraceEvent
from dubai_planner.services.ownership import (
    MIN_TOKEN_LENGTH,
    Owner,
    SessionNotFound,
    new_visitor_token,
)
from dubai_planner.services.runs import PlannerHub, RunManager
from dubai_planner.services.session import SessionService
from dubai_planner.storage import Repository
from dubai_planner.storage.repository import MIGRATIONS, SCHEMA_VERSION
from dubai_planner.tools.toolbox import Toolbox


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "app.db")


def visitor() -> Owner:
    return Owner.from_token(new_visitor_token())


def session_with_content(sessions: SessionService, title: str = "Trip") -> tuple[str, str]:
    """A session of ``sessions.owner`` with a message, two versions and a traced run."""
    sid = sessions.create(PlannerMode.LANGGRAPH, title).id
    repo = sessions.repo
    run = repo.start_run(PlannerMode.LANGGRAPH, session_id=sid)
    repo.add_message(sid, ChatMessage(role=ChatRole.USER, content=f"{title} request",
                                      metadata={"run_id": run.id}))
    repo.add_trace_event(TraceEvent(run_id=run.id, mode=PlannerMode.LANGGRAPH,
                                    agent="parse_request",
                                    event="start", payload={"kind": "node"}))
    repo.save_itinerary(itinerary(sid), request=trip_request(), run_id=run.id)
    repo.save_itinerary(itinerary(sid, days=3), request=trip_request())
    return sid, run.id


# ------------------------------------------------------------------ owner tokens


def test_visitor_tokens_are_random_and_only_their_hash_is_used() -> None:
    a, b = new_visitor_token(), new_visitor_token()
    assert a != b and len(a) >= MIN_TOKEN_LENGTH
    owner = Owner.from_token(a)
    assert owner.key == hashlib.sha256(f"visitor\n{a}".encode()).hexdigest()
    assert a not in str(owner.key) and owner != Owner.from_token(b)
    assert repr(owner) == "Owner(visitor)" and owner.key not in repr(owner)
    assert Owner.LOCAL.is_local and not owner.is_local


@pytest.mark.parametrize("bad", ["", "short", "x" * (MIN_TOKEN_LENGTH - 1), None, 123])
def test_weak_or_missing_tokens_are_rejected(bad: Any) -> None:
    with pytest.raises(ValueError):
        Owner.from_token(bad)


def test_identity_keys_are_stable_and_domain_separated() -> None:
    a = Owner.from_identity("https://accounts.example", "user-1")
    assert a == Owner.from_identity("https://accounts.example", "user-1")
    assert a != Owner.from_identity("https://accounts.example", "user-2")
    with pytest.raises(ValueError):
        Owner.from_identity("", "user-1")


def test_visitor_scope_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SESSION_SCOPE", raising=False)
    assert Settings(_env_file=None).session_scope is SessionScope.VISITOR  # type: ignore[call-arg]


# ------------------------------------------------------------------ two visitors


def test_two_visitors_see_only_their_own_sessions_and_runs(repo: Repository) -> None:
    base = SessionService(repo)
    alice, bob = base.scoped(visitor()), base.scoped(visitor())
    a_sid, a_run = session_with_content(alice, "Alice")
    b_sid, b_run = session_with_content(bob, "Bob")
    assert [s.id for s in alice.list_sessions()] == [a_sid]
    assert [s.id for s in bob.list_sessions()] == [b_sid]
    assert [r.id for r in alice.run_records()] == [a_run]
    assert [r.id for r in bob.run_records()] == [b_run]
    assert alice.owns(a_sid) and not alice.owns(b_sid)
    assert {s.id for s in base.list_sessions()} == {a_sid, b_sid}  # local sees all
    assert len(alice.versions(a_sid)) == 2 and alice.history(a_sid)[0].content == "Alice request"


def test_unauthorized_reads_behave_like_missing_sessions(repo: Repository) -> None:
    base = SessionService(repo)
    alice, mallory = base.scoped(visitor()), base.scoped(visitor())
    sid, run_id = session_with_content(alice)
    assert mallory.get(sid) is None
    for read in (mallory.history, mallory.versions, mallory.latest, mallory.runs,
                 mallory.run_records):
        with pytest.raises(SessionNotFound, match="session not found"):
            read(sid)
    with pytest.raises(SessionNotFound):
        mallory.version(sid, 1)
    missing = pytest.raises(SessionNotFound)  # same error as a foreign session
    with missing:
        mallory.history("does-not-exist")
    assert mallory.get_run(run_id) is None and mallory.trace(run_id) is None
    assert mallory.run_messages(alice.run_records()) == {}
    assert alice.run_messages(alice.run_records()) == {run_id: "Trip request"}


def test_unauthorized_rename_and_delete_change_nothing(repo: Repository) -> None:
    base = SessionService(repo)
    alice, mallory = base.scoped(visitor()), base.scoped(visitor())
    sid, run_id = session_with_content(alice, "Mine")
    with pytest.raises(SessionNotFound):
        mallory.rename(sid, "pwned")
    with pytest.raises(SessionNotFound):
        mallory.delete(sid)
    kept = alice.get(sid)
    assert kept is not None and kept.title == "Mine"
    assert len(alice.versions(sid)) == 2 and repo.get_run(run_id) is not None
    assert alice.delete(sid) and alice.get(sid) is None


def test_repository_repeats_the_owner_condition_in_sql(repo: Repository) -> None:
    owner = visitor()
    sid = SessionService(repo).scoped(owner).create(PlannerMode.LANGGRAPH).id
    other = visitor().key
    assert other is not None
    assert not repo.rename_session(sid, "x", owner=other)
    assert not repo.delete_session(sid, owner=other)
    assert repo.get_session(sid) is not None


def test_unauthorized_export_and_version_comparison(repo: Repository, toolbox: Toolbox
                                                    ) -> None:
    base = SessionService(repo)
    alice, mallory = base.scoped(visitor()), base.scoped(visitor())
    sid, run_id = session_with_content(alice)
    assert alice.compare_versions(sid, 1, 2).new_version == 2
    assert '"version": 1' in alice.export_version(sid, 1, "json", toolbox.catalog)
    assert "Day 1" in alice.export_version(sid, 2, "markdown", toolbox.catalog)
    assert run_id in alice.export_trace(run_id)
    with pytest.raises(SessionNotFound):
        mallory.compare_versions(sid, 1, 2)
    for fmt in ("json", "markdown"):
        with pytest.raises(SessionNotFound):
            mallory.export_version(sid, 1, fmt, toolbox.catalog)  # type: ignore[arg-type]
    with pytest.raises(SessionNotFound):
        mallory.export_trace(run_id)


def test_a_visitor_cannot_claim_or_write_into_a_foreign_session(repo: Repository) -> None:
    base = SessionService(repo)
    alice, mallory = base.scoped(visitor()), base.scoped(visitor())
    sid = alice.create(PlannerMode.LANGGRAPH).id
    with pytest.raises(SessionNotFound):
        mallory.ensure(sid, PlannerMode.LANGGRAPH)
    fresh = mallory.ensure("fresh-session-id", PlannerMode.LANGGRAPH)
    assert mallory.owns(fresh.id) and not alice.owns(fresh.id)


# ------------------------------------------------------------------ runs


@pytest.fixture
def runs(tmp_path: Path, toolbox: Toolbox, repo: Repository) -> Iterator[RunManager]:
    hub = PlannerHub(repo, checkpoint_path=tmp_path / "cp.db", tools=toolbox, llm=MockLLM())
    yield RunManager(hub, SessionService(repo), today=lambda: TODAY)
    hub.close()


def test_run_manager_views_are_isolated_but_share_one_registry(runs: RunManager) -> None:
    alice_s = runs.sessions.scoped(visitor())
    mallory_s = runs.sessions.scoped(visitor())
    alice, mallory = runs.scoped(alice_s), runs.scoped(mallory_s)
    sid = alice_s.create(PlannerMode.LANGGRAPH).id
    handle = alice.start(sid, PlannerMode.LANGGRAPH, WINTER)
    assert handle.wait(120) and handle.error is None
    assert alice.last(sid) is handle and runs.last(sid) is handle  # one registry
    assert mallory.last(sid) is None and mallory.active(sid) is None
    with pytest.raises(SessionNotFound):
        mallory.start(sid, PlannerMode.LANGGRAPH, "make it cheaper")
    with pytest.raises(SessionNotFound):
        mallory.prepare(sid, PlannerMode.LANGGRAPH, "make it cheaper")
    mallory.forget(sid)  # a no-op for a foreign session
    assert alice.last(sid) is handle
    assert len(alice_s.versions(sid)) == 1 and len(alice_s.history(sid)) == 2
    record = alice_s.latest(sid)
    assert record is not None and record.run_id is not None
    assert alice_s.get_run(record.run_id) is not None
    assert mallory_s.get_run(record.run_id) is None


# ------------------------------------------------------------------ migration


def test_v2_database_migrates_and_legacy_sessions_stay_private(tmp_path: Path) -> None:
    """Pre-v3 sessions get owner NULL: local mode keeps them, visitors never see them."""
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        for version, sql in MIGRATIONS[:2]:
            conn.executescript(sql)
            conn.execute(f"PRAGMA user_version = {version}")
        conn.execute("INSERT INTO sessions (id, title, mode, created_at, updated_at) VALUES "
                     "('legacy', 'Old trip', 'langgraph', '2026-11-01T10:00:00+00:00', "
                     "'2026-11-01T10:00:00+00:00')")
    repo = Repository(path)
    assert repo.schema_version() == SCHEMA_VERSION == 3
    with repo.connect() as conn:
        owner = conn.execute("SELECT owner FROM sessions WHERE id = 'legacy'").fetchone()[0]
    assert owner is None
    local, guest = SessionService(repo), SessionService(repo).scoped(visitor())
    assert [s.id for s in local.list_sessions()] == ["legacy"]
    assert guest.list_sessions() == [] and guest.get("legacy") is None
    with pytest.raises(SessionNotFound):
        guest.delete("legacy")
    assert local.get("legacy") is not None


def test_the_database_never_holds_a_raw_visitor_token(repo: Repository) -> None:
    token = new_visitor_token()
    SessionService(repo).scoped(Owner.from_token(token)).create(PlannerMode.LANGGRAPH)
    assert token.encode() not in repo.db_path.read_bytes()
