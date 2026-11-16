"""SQLite repository for sessions, messages, versioned itineraries, runs, traces and evals.

- One short-lived connection per operation (robust with Streamlit threads and with
  OneDrive briefly locking files); foreign keys are enabled on every connection.
- Schema migrations are tracked with ``PRAGMA user_version``.
- Rows are converted to/from Pydantic models at this boundary; JSON columns store
  ``model_dump_json()`` so ``Decimal`` amounts survive exactly.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from dubai_planner.config import Settings, get_settings
from dubai_planner.models.itinerary import Itinerary, PlannerMode
from dubai_planner.models.records import (
    EvalResult,
    EvalRun,
    ItineraryRecord,
    RunRecord,
    RunStatus,
    Session,
    StoredMessage,
)
from dubai_planner.models.request import TripRequest
from dubai_planner.models.trace import ChatMessage, ChatRole, TraceEvent


def _load_migrations() -> list[tuple[int, str]]:
    schema = resources.files("dubai_planner.storage").joinpath("schema.sql")
    return [(1, schema.read_text(encoding="utf-8"))]


MIGRATIONS: list[tuple[int, str]] = _load_migrations()
SCHEMA_VERSION = MIGRATIONS[-1][0]


def _now() -> datetime:
    return datetime.now(UTC)


def _ts(value: datetime) -> str:
    return value.isoformat()


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class NotFoundError(LookupError):
    pass


class Repository:
    def __init__(
        self, db_path: Path | str | None = None, *, settings: Settings | None = None
    ) -> None:
        if db_path is None:
            db_path = (settings or get_settings()).db_path
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    # ------------------------------------------------------------------ infrastructure
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:  # commit on success, rollback on error
                yield conn
        finally:
            conn.close()

    def schema_version(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> int:
        """Apply pending migrations; returns the resulting schema version."""
        current = self.schema_version()
        for version, sql in MIGRATIONS:
            if version > current:
                with self.connect() as conn:
                    conn.executescript(sql)
                    conn.execute(f"PRAGMA user_version = {int(version)}")
                current = version
        return current

    # ------------------------------------------------------------------ sessions
    def create_session(
        self,
        title: str = "New trip",
        mode: PlannerMode = PlannerMode.LANGGRAPH,
        *,
        session_id: str | None = None,
    ) -> Session:
        session = Session(id=session_id or uuid.uuid4().hex, title=title, mode=mode)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, mode, created_at, updated_at) VALUES (?,?,?,?,?)",
                (
                    session.id,
                    session.title,
                    session.mode.value,
                    _ts(session.created_at),
                    _ts(session.updated_at),
                ),
            )
        return session

    def get_session(self, session_id: str) -> Session | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._session(row) if row else None

    def list_sessions(self) -> list[Session]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC, id").fetchall()
        return [self._session(r) for r in rows]

    def rename_session(self, session_id: str, title: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, _ts(_now()), session_id),
            )
        return cur.rowcount == 1

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and (via ON DELETE CASCADE) its messages, runs and itineraries."""
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cur.rowcount == 1

    def _touch(self, conn: sqlite3.Connection, session_id: str) -> None:
        cur = conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (_ts(_now()), session_id)
        )
        if cur.rowcount != 1:
            raise NotFoundError(f"session {session_id!r} does not exist")

    @staticmethod
    def _session(row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            title=row["title"],
            mode=PlannerMode(row["mode"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # ------------------------------------------------------------------ messages
    def add_message(self, session_id: str, message: ChatMessage) -> StoredMessage:
        with self.connect() as conn:
            self._touch(conn, session_id)
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata_json, created_at) "
                "VALUES (?,?,?,?,?)",
                (
                    session_id,
                    message.role.value,
                    message.content,
                    json.dumps(message.metadata, default=str),
                    _ts(message.created_at),
                ),
            )
            msg_id = cur.lastrowid
        assert msg_id is not None  # noqa: S101 - sqlite always returns a rowid on INSERT
        return StoredMessage(
            id=msg_id,
            session_id=session_id,
            role=message.role.value,
            content=message.content,
            created_at=message.created_at,
            metadata=message.metadata,
        )

    def list_messages(self, session_id: str) -> list[StoredMessage]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return [
            StoredMessage(
                id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                created_at=datetime.fromisoformat(r["created_at"]),
                metadata=json.loads(r["metadata_json"]),
            )
            for r in rows
        ]

    def chat_history(self, session_id: str) -> list[ChatMessage]:
        return [
            ChatMessage(
                role=ChatRole(m.role),
                content=m.content,
                created_at=m.created_at,
                metadata=m.metadata,
            )
            for m in self.list_messages(session_id)
        ]

    # ------------------------------------------------------------------ itineraries
    def save_itinerary(
        self,
        itinerary: Itinerary,
        *,
        request: TripRequest | None = None,
        approved: bool = False,
        run_id: str | None = None,
    ) -> ItineraryRecord:
        """Save as the next version of its session; ``parent_id`` links to the previous one.

        The stored itinerary gets a fresh id when the given id already exists, so saving
        an edited copy of v1 never overwrites v1.
        """
        with self.connect() as conn:
            self._touch(conn, itinerary.session_id)
            latest = conn.execute(
                "SELECT id, version FROM itineraries WHERE session_id = ? "
                "ORDER BY version DESC LIMIT 1",
                (itinerary.session_id,),
            ).fetchone()
            exists = conn.execute(
                "SELECT 1 FROM itineraries WHERE id = ?", (str(itinerary.id),)
            ).fetchone()
            update: dict[str, Any] = {
                "version": (latest["version"] + 1) if latest else 1,
                "parent_id": uuid.UUID(latest["id"]) if latest else None,
            }
            if exists:
                update["id"] = uuid.uuid4()
            stored = Itinerary.model_validate({**itinerary.model_dump(), **update})
            created_at = _now()
            conn.execute(
                "INSERT INTO itineraries (id, session_id, version, parent_id, mode, run_id, "
                "request_json, itinerary_json, total_aed, approved, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(stored.id),
                    stored.session_id,
                    stored.version,
                    str(stored.parent_id) if stored.parent_id else None,
                    stored.mode.value,
                    run_id,
                    request.model_dump_json() if request else None,
                    stored.model_dump_json(),
                    float(stored.cost.total),
                    int(approved),
                    _ts(created_at),
                ),
            )
        return ItineraryRecord(
            id=stored.id,
            session_id=stored.session_id,
            version=stored.version,
            parent_id=stored.parent_id,
            mode=stored.mode,
            run_id=run_id,
            total_aed=stored.cost.total,
            approved=approved,
            created_at=created_at,
            request=request,
            itinerary=stored,
        )

    def get_itinerary(self, itinerary_id: str | uuid.UUID) -> ItineraryRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM itineraries WHERE id = ?", (str(itinerary_id),)
            ).fetchone()
        return self._itinerary(row) if row else None

    def latest_itinerary(self, session_id: str) -> ItineraryRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM itineraries WHERE session_id = ? ORDER BY version DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return self._itinerary(row) if row else None

    def list_itineraries(self, session_id: str) -> list[ItineraryRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM itineraries WHERE session_id = ? ORDER BY version", (session_id,)
            ).fetchall()
        return [self._itinerary(r) for r in rows]

    @staticmethod
    def _itinerary(row: sqlite3.Row) -> ItineraryRecord:
        itinerary = Itinerary.model_validate_json(row["itinerary_json"])
        return ItineraryRecord(
            id=uuid.UUID(row["id"]),
            session_id=row["session_id"],
            version=row["version"],
            parent_id=uuid.UUID(row["parent_id"]) if row["parent_id"] else None,
            mode=PlannerMode(row["mode"]),
            run_id=row["run_id"],
            total_aed=itinerary.cost.total,
            approved=bool(row["approved"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            request=(
                TripRequest.model_validate_json(row["request_json"])
                if row["request_json"]
                else None
            ),
            itinerary=itinerary,
        )

    # ------------------------------------------------------------------ runs
    def start_run(
        self, mode: PlannerMode, *, session_id: str | None = None, run_id: str | None = None
    ) -> RunRecord:
        run = RunRecord(id=run_id or uuid.uuid4().hex, session_id=session_id, mode=mode)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO runs (id, session_id, mode, status, started_at) VALUES (?,?,?,?,?)",
                (run.id, run.session_id, run.mode.value, run.status.value, _ts(run.started_at)),
            )
        return run

    def finish_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        llm_calls: int = 0,
        tokens: int = 0,
        error: str | None = None,
    ) -> RunRecord:
        run = self.get_run(run_id)
        if run is None:
            raise NotFoundError(f"run {run_id!r} does not exist")
        ended = _now()
        duration = (ended - run.started_at).total_seconds()
        with self.connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, ended_at = ?, duration_s = ?, llm_calls = ?, "
                "tokens = ?, error = ? WHERE id = ?",
                (status.value, _ts(ended), duration, llm_calls, tokens, error, run_id),
            )
        return run.model_copy(
            update={
                "status": status,
                "ended_at": ended,
                "duration_s": duration,
                "llm_calls": llm_calls,
                "tokens": tokens,
                "error": error,
            }
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._run(row) if row else None

    def list_runs(self, session_id: str | None = None) -> list[RunRecord]:
        with self.connect() as conn:
            if session_id is None:
                rows = conn.execute("SELECT * FROM runs ORDER BY started_at").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE session_id = ? ORDER BY started_at", (session_id,)
                ).fetchall()
        return [self._run(r) for r in rows]

    @staticmethod
    def _run(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=row["id"],
            session_id=row["session_id"],
            mode=PlannerMode(row["mode"]),
            status=RunStatus(row["status"]),
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=_dt(row["ended_at"]),
            duration_s=row["duration_s"],
            llm_calls=row["llm_calls"],
            tokens=row["tokens"],
            error=row["error"],
        )

    # ------------------------------------------------------------------ trace events
    def add_trace_events(self, events: list[TraceEvent]) -> None:
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO trace_events (run_id, agent, event, started_at, ended_at, "
                "tokens_in, tokens_out, payload_json) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        e.run_id,
                        e.agent,
                        e.event.value,
                        _ts(e.started_at),
                        _ts(e.ended_at) if e.ended_at else None,
                        e.tokens_in,
                        e.tokens_out,
                        e.model_dump_json(),
                    )
                    for e in events
                ],
            )

    def add_trace_event(self, event: TraceEvent) -> None:
        self.add_trace_events([event])

    def list_trace_events(self, run_id: str) -> list[TraceEvent]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM trace_events WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return [TraceEvent.model_validate_json(r["payload_json"]) for r in rows]

    # ------------------------------------------------------------------ evaluation
    def create_eval_run(self, config: dict[str, Any] | None = None) -> EvalRun:
        run = EvalRun(id=uuid.uuid4().hex, config=config or {})
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO eval_runs (id, created_at, config_json) VALUES (?,?,?)",
                (run.id, _ts(run.created_at), json.dumps(run.config, default=str)),
            )
        return run

    def add_eval_result(self, result: EvalResult) -> EvalResult:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO eval_results (eval_run_id, scenario_id, mode, seed, metrics_json) "
                "VALUES (?,?,?,?,?)",
                (
                    result.eval_run_id,
                    result.scenario_id,
                    result.mode.value,
                    result.seed,
                    json.dumps(result.metrics, default=str),
                ),
            )
        return result.model_copy(update={"id": cur.lastrowid})

    def list_eval_results(self, eval_run_id: str) -> list[EvalResult]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_results WHERE eval_run_id = ? ORDER BY id", (eval_run_id,)
            ).fetchall()
        return [
            EvalResult(
                id=r["id"],
                eval_run_id=r["eval_run_id"],
                scenario_id=r["scenario_id"],
                mode=PlannerMode(r["mode"]),
                seed=r["seed"],
                metrics=json.loads(r["metrics_json"]),
            )
            for r in rows
        ]


def init_db(settings: Settings | None = None) -> Repository:
    """Create/migrate the application database at ``DB_PATH``."""
    return Repository(settings=settings)
