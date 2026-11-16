-- Schema version 1 (applied when PRAGMA user_version = 0).
-- Timestamps are ISO-8601 UTC strings. JSON columns hold Pydantic model_dump_json() output,
-- so Decimal amounts keep their exact string form; total_aed is a REAL copy for sorting only.

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New trip',
    mode        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role          TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content       TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    session_id  TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    mode        TEXT NOT NULL,
    status      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    duration_s  REAL,
    llm_calls   INTEGER NOT NULL DEFAULT 0,
    tokens      INTEGER NOT NULL DEFAULT 0,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id, started_at);

CREATE TABLE IF NOT EXISTS itineraries (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    version         INTEGER NOT NULL CHECK (version >= 1),
    parent_id       TEXT REFERENCES itineraries(id) ON DELETE SET NULL,
    mode            TEXT NOT NULL,
    run_id          TEXT REFERENCES runs(id) ON DELETE SET NULL,
    request_json    TEXT,
    itinerary_json  TEXT NOT NULL,
    total_aed       REAL NOT NULL,
    approved        INTEGER NOT NULL DEFAULT 0 CHECK (approved IN (0, 1)),
    created_at      TEXT NOT NULL,
    UNIQUE (session_id, version)
);

CREATE TABLE IF NOT EXISTS trace_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    agent         TEXT NOT NULL,
    event         TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    tokens_in     INTEGER NOT NULL DEFAULT 0,
    tokens_out    INTEGER NOT NULL DEFAULT 0,
    payload_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_run ON trace_events(run_id, id);

CREATE TABLE IF NOT EXISTS eval_runs (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    config_json  TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS eval_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_run_id   TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    scenario_id   TEXT NOT NULL,
    mode          TEXT NOT NULL,
    seed          INTEGER NOT NULL DEFAULT 0,
    metrics_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(eval_run_id);
