-- 003: tier routing — the deep-tier daily cap counter.
-- Runs against the aerys_v2 database (NAS Postgres), same home as the checkpointer
-- and outbox. This is V1's `aerys_model_usage` opus cap reborn, minus the
-- check-then-increment race: factory.deep_gate_for spends a credit with ONE atomic
-- statement (INSERT ... ON CONFLICT DO UPDATE ... WHERE call_count < cap RETURNING)
-- — no row returned means the cap held and the turn downgrades to standard.
--
-- One row per (day, tier). Only 'deep' is capped today, but the shape leaves room
-- for rationing any tier without a schema change. day is CURRENT_DATE (UTC — the
-- server's zone), matching the V1 cap's "10/day" semantics.
CREATE TABLE IF NOT EXISTS v2_model_usage (
    day         DATE        NOT NULL,
    tier        TEXT        NOT NULL,
    call_count  INTEGER     NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (day, tier)
);
