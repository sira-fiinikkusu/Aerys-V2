-- 012: Phase 3 in shadow — two privacy verdicts per judged turn, no content.
-- Runs against the aerys_v2 database (NAS Postgres), same as 011.
--
-- WHY: before the content-privacy question can ride the Jev call, we need to know
-- how often Jev agrees with the metered judge on REAL turns. The row keeps the
-- judge's word, whether the keyword pass short-circuited, Jev's p(public), its
-- error or latency, and the sample length. Never the text: this is the judge of
-- what must not be repeated.
CREATE TABLE IF NOT EXISTS v2_privacy_shadow (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    judge           TEXT NOT NULL,           -- 'public' | 'private' (the verdict that acted)
    keyword_hit     BOOLEAN NOT NULL DEFAULT FALSE,
    jev_p_public    REAL,
    jev_error       TEXT,
    jev_latency_ms  INTEGER,
    sample_len      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS v2_privacy_shadow_created_idx ON v2_privacy_shadow (created_at DESC);
