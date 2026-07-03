-- 001: the migration spine — turns audit + action outbox.
-- Runs against the aerys database (NAS Postgres). Applied in Phase 2 alongside the
-- LangGraph checkpointer tables; committed now because every later component writes
-- through these shapes and earlier canaries must not invent incompatible locals.
--
-- Design doc: docs/design/2026-07-02-turns-outbox-spine.md

-- ---------------------------------------------------------------- turns (audit)
-- One row per ask() turn. The LangGraph checkpointer stays messages-only; THIS is
-- where forensics live: who was resolved, what tier fired, raw vs polished output,
-- how delivery went. Append-only; never read on the hot path.
CREATE TABLE IF NOT EXISTS v2_turns (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    thread_id       TEXT        NOT NULL,           -- checkpointer thread key
    channel         TEXT        NOT NULL,           -- 'discord_dm' | 'guild' | 'voice' | 'cli' | ...
    -- identity snapshot (auditability without checkpointing identity):
    person_id       UUID,
    platform_identity TEXT,                          -- e.g. 'discord:60426...'
    resolver_version TEXT,
    -- routing decision (reproducibility for evals/incidents):
    classifier_intent TEXT,
    tier            TEXT,
    tier_override_source TEXT,                       -- null | 'owner_phrase' | 'fallback_retry'
    guard_verdict   TEXT,
    -- content provenance (polish may never overwrite the only copy):
    input_text      TEXT        NOT NULL,
    raw_reply       TEXT,                            -- model output before polish
    emitted_reply   TEXT,                            -- what actually went to the channel
    -- health:
    tool_calls      JSONB       NOT NULL DEFAULT '[]'::jsonb,
    degraded        JSONB       NOT NULL DEFAULT '[]'::jsonb,  -- e.g. ["ha_unreachable"]
    error           TEXT,
    latency_ms      INTEGER,
    trace_id        TEXT
);
CREATE INDEX IF NOT EXISTS v2_turns_thread_idx  ON v2_turns (thread_id, created_at);
CREATE INDEX IF NOT EXISTS v2_turns_person_idx  ON v2_turns (person_id, created_at);

-- ---------------------------------------------------------------- outbox (actions)
-- Write-ahead intent for EVERY side effect: message delivery, HA writes, governance
-- writes, proactive sends, email. Pattern: INSERT intent -> execute -> record result.
-- Crash between steps = a pending row a sweeper can reconcile, never a duplicate or
-- silent loss. idempotency_key makes retries safe (executor checks before firing).
CREATE TABLE IF NOT EXISTS v2_outbox (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    turn_id         BIGINT REFERENCES v2_turns(id),
    kind            TEXT        NOT NULL,            -- 'emit' | 'ha_write' | 'governance' | 'email' | ...
    payload         JSONB       NOT NULL,
    idempotency_key TEXT        NOT NULL UNIQUE,
    -- confirmation semantics (owner-gated / HITL actions park here until confirmed):
    requires_confirmation BOOLEAN NOT NULL DEFAULT FALSE,
    confirmation_binding  TEXT,                      -- person_id that must say yes
    status          TEXT        NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','awaiting_confirmation','executing',
                                      'succeeded','failed','cancelled','expired')),
    attempts        INTEGER     NOT NULL DEFAULT 0,
    last_error      TEXT,
    receipt         JSONB,                           -- e.g. {"message_id": "..."} from Discord
    expires_at      TIMESTAMPTZ                      -- confirmations rot; expired != cancelled
);
CREATE INDEX IF NOT EXISTS v2_outbox_status_idx ON v2_outbox (status, created_at);

-- ---------------------------------------------------------------- writer lease
-- Mechanical "one armed writer" (cross-review #11): before firing, a write capability
-- SELECTs its kind's lease and refuses unless holder matches itself. Cutover = one
-- UPDATE flipping the holder from 'n8n' to 'brain'; rollback = the same UPDATE back.
CREATE TABLE IF NOT EXISTS v2_writer_lease (
    kind            TEXT PRIMARY KEY,                -- matches v2_outbox.kind
    holder          TEXT NOT NULL,                   -- 'n8n' | 'brain'
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    note            TEXT
);
INSERT INTO v2_writer_lease (kind, holder, note) VALUES
    ('emit',       'n8n', 'flips per-channel during Wave 3-5 cutovers'),
    ('ha_write',   'n8n', ''),
    ('governance', 'n8n', ''),
    ('email',      'n8n', 'dead last per dossier sequencing')
ON CONFLICT (kind) DO NOTHING;
