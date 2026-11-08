-- 015: Kael tickets (board #176, Stage 0). The owner, 2026-11-07: "when i tell her to ask
-- you something, she never follows up with me. the conversation is 1 way."
-- One row per ask she sends down her line with message_kael. The ticket id rides in the
-- message, Kael answers BY TICKET (POST /kael-ticket/{id}), the answer lands in the thread
-- she asked from, and when the owner was the one asking she brings it back to him.
-- Statuses are A2A task states. The ask and the answer are the private part: never logged.
--   outbox        the text still owed to Kael's line (NULL = nothing owed; the watcher resends)
--   waiting_since when it last started waiting on Kael (an update from her resets it)
--   answer        Kael's words that go back to the owner (input-required, completed, failed);
--                 answered_at is when, relayed_at is when she passed them on
--   progress      Kael's words while working (she sees them; they never replace an answer)
CREATE TABLE IF NOT EXISTS v2_kael_tickets (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    thread_id     TEXT NOT NULL,
    ask           TEXT NOT NULL CHECK (length(btrim(ask)) > 0),
    for_owner     BOOLEAN NOT NULL DEFAULT FALSE,
    status        TEXT NOT NULL DEFAULT 'submitted' CHECK (status IN
                      ('submitted', 'working', 'input-required', 'completed', 'failed', 'canceled')),
    outbox        TEXT,
    last_try_at   TIMESTAMPTZ,
    delivered_at  TIMESTAMPTZ,
    waiting_since TIMESTAMPTZ NOT NULL DEFAULT now(),
    answer        TEXT,
    answered_at   TIMESTAMPTZ,
    progress      TEXT,
    relayed_at    TIMESTAMPTZ,
    nagged_at     TIMESTAMPTZ,
    escalated_at  TIMESTAMPTZ
);

-- The watcher reads the open ones (and answers still owed to the owner) every minute.
CREATE INDEX IF NOT EXISTS idx_kael_tickets_open
    ON v2_kael_tickets (created_at)
    WHERE status IN ('submitted', 'working', 'input-required');
