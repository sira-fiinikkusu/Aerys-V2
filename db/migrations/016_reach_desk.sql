-- 016: the reach-out desk (board #176, Stage 2). Every message she starts herself goes
-- through one desk with its checks in code: no repeats, the night, his meetings, a daily
-- budget he steers in his own words, topics he muted. This is its record: what she meant
-- to say, whether it went, where, and why it waited or didn't go. The owner ruled D4-D7
-- on 2026-11-08 (adaptive budget, presence-led rooms, natural-language feedback).
-- headline is what may be spoken aloud; detail only ever goes as text. Both are the
-- private part: never logged (logs carry ids, sources and statuses).
CREATE TABLE IF NOT EXISTS v2_reachouts (
    id           BIGSERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    source       TEXT NOT NULL,
    topic        TEXT NOT NULL,
    urgency      TEXT NOT NULL CHECK (urgency IN ('routine', 'timely', 'safety')),
    owed         BOOLEAN NOT NULL DEFAULT FALSE,   -- an answer he asked for: never dropped
    headline     TEXT NOT NULL CHECK (length(btrim(headline)) > 0),
    detail       TEXT NOT NULL DEFAULT '',
    dedupe_key   TEXT,
    status       TEXT NOT NULL CHECK (status IN ('queued', 'sent', 'dropped', 'failed')),
    not_before   TIMESTAMPTZ,                      -- a queued item's window opens here
    expires_at   TIMESTAMPTZ,
    reason       TEXT NOT NULL DEFAULT '',         -- why it waited, or didn't go
    surface      TEXT,
    silent       BOOLEAN NOT NULL DEFAULT FALSE,
    spoken_at    TIMESTAMPTZ,
    dm_at        TIMESTAMPTZ,                      -- the text copy (detail, or "in case you missed it")
    delivered_at TIMESTAMPTZ,
    acked_at     TIMESTAMPTZ,                      -- he answered after it was spoken: no copy needed
    attempts     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_reachouts_queued ON v2_reachouts (not_before) WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS idx_reachouts_dedupe ON v2_reachouts (dedupe_key) WHERE dedupe_key IS NOT NULL;

-- What he's told her he wants: one row. budget = unsolicited reach-outs a day (the briefs,
-- answers he asked for and safety warnings don't count). muted = topics he said to stop.
CREATE TABLE IF NOT EXISTS v2_reach_prefs (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    budget     INTEGER NOT NULL CHECK (budget BETWEEN 0 AND 8),
    muted      TEXT[] NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every change to those, with his words, so "why did she go quiet?" has an answer.
CREATE TABLE IF NOT EXISTS v2_reach_pref_changes (
    id     BIGSERIAL PRIMARY KEY,
    at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    change TEXT NOT NULL,
    words  TEXT NOT NULL DEFAULT ''
);
