-- 022: the owner's approval button (board #176, Stage 4's shared piece). Some things she may only
-- do once he taps Approve on a Discord button: D10 (11/08) "Voice can start anything text can but
-- requires me to approve via discord button"; D12 (11/09) the first message to anyone new waits
-- for his tap, and the button shows exactly who and where. One row per request; the gateway posts
-- the button, his tap decides it, and anything but an explicit approve from him in time is a no.
-- summary/detail are what he's approving: the private part, never logged.
CREATE TABLE IF NOT EXISTS v2_approvals (
    id           BIGSERIAL PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (length(btrim(kind)) > 0),
    summary      TEXT NOT NULL CHECK (length(btrim(summary)) > 0),
    detail       TEXT NOT NULL DEFAULT '',
    requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at   TIMESTAMPTZ NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'approved', 'declined', 'expired', 'withdrawn')),
    decided_at   TIMESTAMPTZ,
    decided_by   TEXT,                             -- the Discord user id that tapped (only ever his)
    request_key  TEXT NOT NULL UNIQUE,             -- asking twice for the same thing is one request
    message_id   TEXT,                             -- the DM carrying the buttons ('posting:<token>' while claimed)
    shown        TEXT,                             -- the status that DM last showed (edits retried until it matches)
    pinged_at    TIMESTAMPTZ                       -- when the HA notification went (retried until it does)
);

CREATE INDEX IF NOT EXISTS idx_approvals_pending ON v2_approvals (requested_at) WHERE status = 'pending';
