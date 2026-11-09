-- 023: what waits for his tap (board #176, Stage 4). D10 (11/08): from voice she may start
-- read-only home jobs on a strong match; anything else waits for his Discord button. D12 (11/09):
-- the first message to anyone new waits for his button, which shows exactly who and where. The
-- button is v2_approvals (022); this is the action behind it: what to do once he taps Approve,
-- done once and only then. A declined, expired or withdrawn request is dropped, never done.
-- args are the private part (the job's words, the message): never logged.
CREATE TABLE IF NOT EXISTS v2_gated_actions (
    id          BIGSERIAL PRIMARY KEY,
    approval_id BIGINT NOT NULL UNIQUE REFERENCES v2_approvals (id),
    action      TEXT NOT NULL CHECK (action IN ('home_job', 'message')),
    base_key    TEXT NOT NULL,                     -- the same thing asked again: one attempt per decision
    attempt     INTEGER NOT NULL DEFAULT 1 CHECK (attempt >= 1),
    thread_id   TEXT NOT NULL,                     -- where she asked from (the outcome is noted there)
    args        JSONB NOT NULL,
    status      TEXT NOT NULL DEFAULT 'waiting'
                CHECK (status IN ('waiting', 'running', 'done', 'failed', 'dropped')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    outcome     TEXT,
    UNIQUE (base_key, attempt)
);

CREATE INDEX IF NOT EXISTS idx_gated_actions_waiting ON v2_gated_actions (id) WHERE status = 'waiting';
CREATE INDEX IF NOT EXISTS idx_gated_actions_key ON v2_gated_actions (base_key);
