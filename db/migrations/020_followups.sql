-- 020: follow-ups (board #176, Stage 2). The owner, 2026-11-08, picked C: she offers "want me
-- to check back on that?" and checks back when he says yes; unasked, she checks in only about
-- appointments on his personal calendar and things he said he'd do, a few a day at most, and
-- he steers that in his own words ("is adaptability an option with this?").
--   asked        he said yes (or asked her to check on him): owed, never dropped
--   appointment  a personal calendar event that just ended: "how'd it go?"
--   plan         something he said he'd do, mined from what he told her: SHADOW first (logged,
--                never said) until its extractions have been read
-- about/his_words are the private part: never logged.
CREATE TABLE IF NOT EXISTS v2_followups (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    kind        TEXT NOT NULL CHECK (kind IN ('asked', 'appointment', 'plan')),
    about       TEXT NOT NULL CHECK (length(btrim(about)) > 0),
    his_words   TEXT NOT NULL DEFAULT '',
    due_at      TIMESTAMPTZ NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('waiting', 'raised', 'canceled', 'skipped', 'shadow')),
    source_key  TEXT UNIQUE,                       -- one follow-up per calendar event / mined turn
    reason      TEXT NOT NULL DEFAULT ''           -- what became of it, in a few words
);

CREATE INDEX IF NOT EXISTS idx_followups_waiting ON v2_followups (due_at) WHERE status = 'waiting';

-- Where the plan miner has read up to (a v2_turns id), so a restart never re-reads.
CREATE TABLE IF NOT EXISTS v2_followup_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- How many unasked check-ins a day he's said he wants (0-4): "check on me more/less".
ALTER TABLE v2_reach_prefs ADD COLUMN IF NOT EXISTS checkins INTEGER NOT NULL DEFAULT 2
    CHECK (checkins BETWEEN 0 AND 4);
