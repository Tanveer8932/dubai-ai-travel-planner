"""AC1.5: repository CRUD, itinerary versioning, and persistence across process restarts."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from factories import itinerary, trip_request

from dubai_planner.config import PROJECT_ROOT
from dubai_planner.models import (
    ChatMessage,
    ChatRole,
    EvalResult,
    PlannerMode,
    RunStatus,
    TraceEvent,
    TraceEventType,
)
from dubai_planner.storage import NotFoundError, Repository
from dubai_planner.storage.repository import SCHEMA_VERSION


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "app.db")


def test_migration_sets_user_version_and_is_idempotent(repo: Repository) -> None:
    assert repo.schema_version() == SCHEMA_VERSION == 1
    assert repo.migrate() == 1
    again = Repository(repo.db_path)  # re-opening an existing DB does not re-apply
    assert again.schema_version() == 1


def test_session_crud(repo: Repository) -> None:
    s = repo.create_session("Family trip", PlannerMode.CREWAI_SEQUENTIAL)
    assert repo.get_session(s.id) == s
    assert repo.rename_session(s.id, "Family trip (Dec)")
    renamed = repo.get_session(s.id)
    assert renamed is not None and renamed.title == "Family trip (Dec)"
    assert renamed.updated_at >= s.updated_at
    other = repo.create_session("Solo")
    assert {x.id for x in repo.list_sessions()} == {s.id, other.id}
    assert repo.delete_session(s.id)
    assert repo.get_session(s.id) is None
    assert not repo.delete_session(s.id)
    assert not repo.rename_session("missing", "x")


def test_messages_round_trip_in_order(repo: Repository) -> None:
    s = repo.create_session()
    repo.add_message(s.id, ChatMessage(role=ChatRole.USER, content="4 days, AED 3000"))
    repo.add_message(
        s.id,
        ChatMessage(
            role=ChatRole.ASSISTANT, content="Here's a plan", metadata={"itinerary_version": 1}
        ),
    )
    history = repo.chat_history(s.id)
    assert [m.role for m in history] == [ChatRole.USER, ChatRole.ASSISTANT]
    assert history[1].metadata == {"itinerary_version": 1}
    with pytest.raises(NotFoundError):
        repo.add_message("missing", ChatMessage(role=ChatRole.USER, content="x"))


def test_itinerary_versions_link_to_parent(repo: Repository) -> None:
    s = repo.create_session()
    req = trip_request()
    v1 = repo.save_itinerary(itinerary(s.id), request=req, approved=True)
    assert v1.version == 1 and v1.parent_id is None and v1.approved

    cheaper = itinerary(s.id, total="900")
    req2 = req.model_copy(update={"budget_aed": Decimal("4800")})
    v2 = repo.save_itinerary(cheaper, request=req2)
    assert v2.version == 2 and v2.parent_id == v1.id
    assert v2.itinerary.version == 2 and v2.itinerary.parent_id == v1.id

    latest = repo.latest_itinerary(s.id)
    assert latest is not None and latest.id == v2.id
    assert latest.total_aed == Decimal("900.00")
    assert latest.request is not None and latest.request.budget_aed == Decimal("4800.00")
    assert [r.version for r in repo.list_itineraries(s.id)] == [1, 2]
    fetched = repo.get_itinerary(v1.id)
    assert fetched is not None and fetched.itinerary == v1.itinerary  # exact Decimal round trip


def test_saving_an_edited_copy_never_overwrites(repo: Repository) -> None:
    s = repo.create_session()
    v1 = repo.save_itinerary(itinerary(s.id))
    v2 = repo.save_itinerary(v1.itinerary)  # same id passed again
    assert v2.id != v1.id and v2.parent_id == v1.id
    assert repo.get_itinerary(v1.id) is not None


def test_itinerary_requires_existing_session(repo: Repository) -> None:
    with pytest.raises(NotFoundError):
        repo.save_itinerary(itinerary("no-such-session"))


def test_versions_are_per_session(repo: Repository) -> None:
    a, b = repo.create_session(), repo.create_session()
    repo.save_itinerary(itinerary(a.id))
    rb = repo.save_itinerary(itinerary(b.id))
    assert rb.version == 1 and rb.parent_id is None


def test_runs_and_trace_events(repo: Repository) -> None:
    s = repo.create_session()
    run = repo.start_run(PlannerMode.LANGGRAPH, session_id=s.id)
    t0 = datetime(2026, 11, 15, 10, tzinfo=UTC)
    events = [
        TraceEvent(
            run_id=run.id,
            mode="langgraph",
            agent=f"research_{n}",
            event=TraceEventType.END,
            started_at=t0,
            ended_at=t0 + timedelta(seconds=1),
            tokens_in=100,
            tokens_out=20,
            payload={"n": n},
        )
        for n in ("attractions", "dining")
    ]
    repo.add_trace_events(events)
    repo.add_trace_event(
        TraceEvent(run_id=run.id, mode="langgraph", agent="budget", event=TraceEventType.START)
    )
    stored = repo.list_trace_events(run.id)
    assert stored[:2] == events and stored[2].agent == "budget"

    done = repo.finish_run(run.id, RunStatus.SUCCEEDED, llm_calls=7, tokens=1234)
    assert done.status is RunStatus.SUCCEEDED and done.duration_s is not None
    assert repo.get_run(run.id) == done
    assert [r.id for r in repo.list_runs(s.id)] == [run.id] == [r.id for r in repo.list_runs()]
    with pytest.raises(NotFoundError):
        repo.finish_run("missing", RunStatus.FAILED)


def test_itinerary_links_to_run(repo: Repository) -> None:
    s = repo.create_session()
    run = repo.start_run(PlannerMode.LANGGRAPH, session_id=s.id)
    rec = repo.save_itinerary(itinerary(s.id), run_id=run.id)
    fetched = repo.get_itinerary(rec.id)
    assert fetched is not None and fetched.run_id == run.id


def test_delete_session_cascades(repo: Repository) -> None:
    s = repo.create_session()
    repo.add_message(s.id, ChatMessage(role=ChatRole.USER, content="hi"))
    run = repo.start_run(PlannerMode.LANGGRAPH, session_id=s.id)
    repo.add_trace_event(TraceEvent(run_id=run.id, mode="m", agent="a", event=TraceEventType.INFO))
    repo.save_itinerary(itinerary(s.id), run_id=run.id)
    repo.delete_session(s.id)
    with sqlite3.connect(repo.db_path) as conn:
        counts = [
            conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608
            for t in ("messages", "runs", "itineraries", "trace_events")
        ]
    assert counts == [0, 0, 0, 0]


def test_foreign_keys_enforced(repo: Repository) -> None:
    with pytest.raises(sqlite3.IntegrityError), repo.connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            ("ghost", "user", "x", "2026-01-01T00:00:00+00:00"),
        )


def test_eval_tables(repo: Repository) -> None:
    run = repo.create_eval_run({"provider": "mock", "seeds": 3})
    saved = repo.add_eval_result(
        EvalResult(
            eval_run_id=run.id,
            scenario_id="family_dec",
            mode=PlannerMode.LANGGRAPH,
            seed=1,
            metrics={"budget_ok": True, "latency_s": 1.5},
        )
    )
    assert saved.id is not None
    assert repo.list_eval_results(run.id) == [saved]


def test_data_survives_process_restart(tmp_path: Path) -> None:
    """Write in this process, read back in a brand-new Python process (file DB)."""
    db = tmp_path / "persist.db"
    repo = Repository(db)
    s = repo.create_session("Persisted")
    repo.add_message(s.id, ChatMessage(role=ChatRole.USER, content="Plan my trip"))
    v1 = repo.save_itinerary(itinerary(s.id), request=trip_request())
    v2 = repo.save_itinerary(itinerary(s.id, total="900"))
    del repo

    code = textwrap.dedent(f"""
        import sys
        from dubai_planner.storage import Repository
        repo = Repository(r"{db}")
        s = repo.get_session("{s.id}")
        msgs = repo.list_messages("{s.id}")
        latest = repo.latest_itinerary("{s.id}")
        assert s.title == "Persisted", s
        assert [m.content for m in msgs] == ["Plan my trip"], msgs
        assert latest.version == 2 and str(latest.parent_id) == "{v1.id}", latest
        assert str(latest.id) == "{v2.id}"
        assert str(latest.total_aed) == "900.00"
        print("OK")
    """)
    result = subprocess.run(  # noqa: S603 - fixed interpreter and inline test code
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
