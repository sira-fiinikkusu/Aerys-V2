-- 008: the Kael line's return direction (task #66, owner-approved 2026-08-04).
-- One row per note Kael injects into a thread via POST /kael-note. The receipt
-- half of the family-circle design: the owner can dig every note, and the
-- family_visible flag marks which exchanges his own threads may splice as
-- shared context (default FALSE — "tests do poison memory").
CREATE TABLE IF NOT EXISTS v2_kael_notes (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    thread_id   TEXT NOT NULL,
    note        TEXT NOT NULL,
    family_visible BOOLEAN NOT NULL DEFAULT FALSE
);

-- The family splice reads newest-first, visible-only.
CREATE INDEX IF NOT EXISTS idx_kael_notes_family
    ON v2_kael_notes (family_visible, created_at DESC);
