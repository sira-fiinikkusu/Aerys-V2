-- 030: her own wake-ups (D40/D41, 2026-11-12). She sets herself a one-off or recurring wake-up
-- for any task that needs checking back on her own (a flight, a delivery, a price, a long job
-- at home); when one fires she runs ONE action turn with her tools, unattended, and tells him
-- (an owed answer through the reach desk), stays quiet, or ends it. Rails (Settings): at most
-- 10 active, fires at least 5 minutes apart, a 7-day default end and a 90-day maximum.
--   note        her words to her future self, written by an attended turn at add; NEVER updated
--               afterwards (wakeups.py's stores refuse it): a wake turn can't rewrite its own orders
--   claimed_until  a runner's lease: a wake-up is claimed before it runs, so a slow run is never
--               started twice, by this brain or another; claim_token says whose lease it is, and only
--               that holder renews or releases it
--   run_log     the last 10 runs: time, outcome, her summary (<= 600 chars), the tools she used.
--               Fed back to the next run as untrusted data, never as instructions.
-- note and run_log are the private part: never logged (ids, statuses and outcomes only).
CREATE TABLE IF NOT EXISTS v2_wakeups (
    id                   BIGSERIAL PRIMARY KEY,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    thread_id            TEXT NOT NULL,                       -- where it was set
    note                 TEXT NOT NULL CHECK (length(btrim(note)) > 0 AND length(note) <= 1000),
    kind                 TEXT NOT NULL CHECK (kind IN ('once', 'recurring')),
    at                   TIMESTAMPTZ,                         -- once
    cron                 TEXT,                                -- recurring: 5 fields, read in tz
    tz                   TEXT NOT NULL DEFAULT 'America/New_York',
    until                TIMESTAMPTZ NOT NULL,
    next_run_at          TIMESTAMPTZ,
    status               TEXT NOT NULL DEFAULT 'active'
                         CHECK (status IN ('active', 'paused', 'ended', 'missed')),
    paused_reason        TEXT NOT NULL DEFAULT '',
    runs                 INTEGER NOT NULL DEFAULT 0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_run_at          TIMESTAMPTZ,
    claimed_until        TIMESTAMPTZ,
    claim_token          TEXT,                                -- whose lease it is: renew/release need it
    run_log              JSONB NOT NULL DEFAULT '[]'::jsonb,
    CHECK ((kind = 'once' AND at IS NOT NULL) OR (kind = 'recurring' AND cron IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS idx_wakeups_due ON v2_wakeups (status, next_run_at);
