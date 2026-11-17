-- Schema version 2 (Phase 6): per-run token/latency aggregation and run context.
-- tokens keeps the v1 meaning (tokens_in + tokens_out). context_json records what a
-- fair comparison needs (provider, model, seed-data fingerprint, loop caps); it never
-- holds secrets.

ALTER TABLE runs ADD COLUMN tokens_in     INTEGER NOT NULL DEFAULT 0;
ALTER TABLE runs ADD COLUMN tokens_out    INTEGER NOT NULL DEFAULT 0;
ALTER TABLE runs ADD COLUMN llm_latency_s REAL    NOT NULL DEFAULT 0;
ALTER TABLE runs ADD COLUMN iterations    INTEGER;
ALTER TABLE runs ADD COLUMN context_json  TEXT    NOT NULL DEFAULT '{}';
