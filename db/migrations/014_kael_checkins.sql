-- 014: Kael check-ins (owner ask 2026-10-12). A recurring job Kael runs for the
-- household lives here instead of in Kael's session crons (which expire weekly):
-- at each check-in's days + local time, the brain's check-in watcher taps Kael's
-- session down her line, and stopping one is as easy as telling her. Rows live only
-- in this table, never in the repo: a name is a generic label (it shows in her list,
-- in logs and in the tap); the message is the private part and is never logged.
-- last_fired / last_skipped are the scheduled LOCAL dates already handled: a
-- check-in fires at most once per scheduled day, and a missed day is reported once.
-- resumed_at: occurrences that fell due while it was off never existed for it.
CREATE TABLE IF NOT EXISTS v2_kael_checkins (
    name         TEXT PRIMARY KEY CHECK (length(btrim(name)) > 0),
    days         TEXT[] NOT NULL CHECK (cardinality(days) > 0
                     AND days <@ ARRAY['mon','tue','wed','thu','fri','sat','sun']),
    at_time      TEXT NOT NULL CHECK (at_time ~ '^([01][0-9]|2[0-3]):[0-5][0-9]$'),
    message      TEXT NOT NULL CHECK (length(btrim(message)) > 0),
    active       BOOLEAN NOT NULL DEFAULT true,
    last_fired   DATE,
    last_skipped DATE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    resumed_at   TIMESTAMPTZ,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
