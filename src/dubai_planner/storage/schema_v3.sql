-- Schema version 3 (deployment security): session ownership.
-- owner holds an opaque owner key: the SHA-256 of a visitor's random token (or of an
-- authenticated identity), never the token itself. Sessions created before v3 keep
-- owner = NULL ("legacy"): they stay visible in SESSION_SCOPE=shared (local single-user
-- mode) and are never shown to, or modifiable by, a visitor in SESSION_SCOPE=visitor.

ALTER TABLE sessions ADD COLUMN owner TEXT;
CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions(owner, updated_at);
