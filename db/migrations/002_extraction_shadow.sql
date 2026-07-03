-- 002: extraction shadow — staging memories + high-water marks.
-- Runs against the aerys_v2 database (NAS Postgres), NOT prod aerys. This is the
-- shadow half of the memory pipeline: the V2 extraction worker (workers/extraction.py)
-- reads prod conversations READ-ONLY and writes ONLY here, so its output can be
-- diffed against what the live n8n batch extraction (IfqY4BrhBGeQrcTC) wrote to
-- prod memories — before the writer lease ever flips.

-- pgvector wasn't installed in aerys_v2 yet (only prod aerys had it). Needed so the
-- staging embedding column matches prod's vector(1536) shape byte-for-byte.
-- pgvector >= 0.7 is a trusted extension — the db owner (sira) may create it.
CREATE EXTENSION IF NOT EXISTS vector;

-- ------------------------------------------------------- v2_memories_staging
-- Mirrors prod `memories` column-for-column (same names/types, minus prod's unused
-- legacy columns: summary, category, channel, batch_job_id, processed_at,
-- source_message_id — the n8n insert never populated them either), plus two
-- shadow-only provenance columns:
--   source_thread  — where the conversation came from ('v1:n8n_chat_histories'
--                    for prod channels, or the v2_turns thread_id for her own)
--   shadow_run_at  — when THIS worker run landed the row (created_at stays the
--                    original message time — the h.created_at-not-NOW() lesson)
-- Append-only: shadow mode does no dedup/soft-delete-replace (that's the prod
-- writer's job); comparing raw extractions is the whole point of staging.
CREATE TABLE IF NOT EXISTS v2_memories_staging (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id       UUID NOT NULL,
    content         TEXT NOT NULL,
    key_label       TEXT,
    context         TEXT,
    event_date      TEXT,
    embedding       VECTOR(1536),
    source_platform TEXT,
    privacy_level   TEXT NOT NULL DEFAULT 'public',
    created_at      TIMESTAMPTZ NOT NULL,            -- original message time, NOT now()
    deleted_at      TIMESTAMPTZ,                     -- kept for shape parity with prod
    source_thread   TEXT NOT NULL,
    shadow_run_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS v2_memories_staging_person_idx
    ON v2_memories_staging (person_id, created_at);
CREATE INDEX IF NOT EXISTS v2_memories_staging_run_idx
    ON v2_memories_staging (shadow_run_at);

-- --------------------------------------------------- v2_extraction_watermark
-- High-water mark per conversation source. last_processed_at is TEXT on purpose:
-- it stores the RAW Postgres timestamp string (created_at::text) verbatim, never a
-- reformatted datetime. The n8n version stored new Date().toISOString() in
-- staticData — JS truncates to milliseconds while timestamptz keeps microseconds,
-- so `>` re-matched the same row forever (worked around with a +1ms bump hack).
-- Raw string round-trips exactly through `> $1::timestamptz`, deleting the bug
-- AND the hack. One row per source ('prod_chat', 'v2_turns').
CREATE TABLE IF NOT EXISTS v2_extraction_watermark (
    source            TEXT PRIMARY KEY,
    last_processed_at TEXT NOT NULL,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
