-- 021: "I'm done" from her home body reaches the owner (board #176, Stage 3). Long jobs run on
-- her home hands body (the Agent Protocol server in aerys-portable's home/server.py); when one
-- ends it POSTs a done-notice here, at least once, from a durable outbox. This is the brain's
-- side: every notice is written down the moment it lands (one row per run and status, so a
-- resend is the same row), answered at once, and handed to the reach-out desk by a worker:
-- he asked for the job, so the answer is owed (never dropped, never budgeted; quiet text at
-- night). metadata is what the brain gave the job's thread; the job's words never land here.
CREATE TABLE IF NOT EXISTS v2_home_notices (
    id          BIGSERIAL PRIMARY KEY,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_id      TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('success', 'error', 'cancelled', 'interrupted', 'timeout')),
    metadata    JSONB NOT NULL DEFAULT '{}',
    attempts    INTEGER NOT NULL DEFAULT 0,   -- passes that couldn't read the job's details yet
    handled_at  TIMESTAMPTZ,                  -- handed to the desk (or nothing to say)
    outcome     TEXT NOT NULL DEFAULT '',     -- the desk's answer, or why nothing was said
    UNIQUE (run_id, status)
);

CREATE INDEX IF NOT EXISTS idx_home_notices_open ON v2_home_notices (id) WHERE handled_at IS NULL;
