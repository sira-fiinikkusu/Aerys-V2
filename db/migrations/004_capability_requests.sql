-- 004: capability-request loop (self-iteration) — the gap-mining spine.
-- Runs against the aerys_v2 database (NAS Postgres), same home as the checkpointer,
-- outbox, model-usage cap, and the v2_turns audit spine this feature MINES. Forward
-- only, non-destructive: every statement is IF NOT EXISTS / ON CONFLICT DO NOTHING,
-- and nothing here DROPs or DELETEs an existing row — a re-run is a no-op.
--
-- Design doc: docs/capability-request-loop-design.md (v3, provenance-tiered).
-- Phase A: the miner (workers/capability_requests.py) reads v2_turns for capability-
-- gap signals and writes ONLY these two tables. It never touches service.py's ask()
-- loop, and it never self-grants: the brain never writes here, the worker writes only
-- observation fields, and approval stays the owner's manual /approve gate (Phase B).

-- ------------------------------------------------------- v2_capability_requests
-- One row per DISTINCT gap fingerprint. `origin_class` is MACHINE-SET by which
-- detector fired and is the whole security model: 'error' comes ONLY from a
-- structural signal the model cannot fake (a `degraded` marker or a real
-- `tool_calls` failure — see turns.py); 'complaint' comes from fakeable,
-- model-authored reply-phrase text and is FORCED onto the stricter approval tier.
-- An injected complaint can never present as an 'error' — the attacker may author
-- the text, but not upgrade its trust level. Mined content is DATA, never
-- instructions: nothing in `summary`/`diagnosis` is ever executed or obeyed.
CREATE TABLE IF NOT EXISTS v2_capability_requests (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    fingerprint     TEXT NOT NULL UNIQUE,           -- bound param, never string-interpolated
    signal_kind     TEXT NOT NULL CHECK (signal_kind IN ('degraded','tool_error','reply_phrase')),
    origin_class    TEXT NOT NULL CHECK (origin_class IN ('error','complaint')), -- MACHINE-SET
    -- summary: FIXED TEMPLATE for errors ("tool 'x' failed (timeout)"); a bounded,
    -- sanitized excerpt of her reply for complaints (that excerpt IS the value —
    -- the owner needs to see what she wished for). Rendered only under the
    -- untrusted-data fence + the "complaint, not an error" label at surfacing time.
    summary         TEXT NOT NULL,
    origin_trust    TEXT NOT NULL DEFAULT 'owner',  -- only owner/allowlisted turns are mined in Phase A
    how_often       INTEGER NOT NULL DEFAULT 1,     -- COUNT(*) over the examples child, never a blind +1
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Kael/owner workflow columns (NEVER written by the miner; reserved for the
    -- Phase-B diagnose/propose/approve path — the brain never self-grants):
    scope_tier      INTEGER,
    diagnosis       TEXT,                           -- Kael's independent read of the raw turns (under the fence)
    proposal        TEXT,
    status          TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open','surfaced','diagnosing','proposed',
                                      'approved','building','built','rejected','wont_fix')),
    -- required_tier is DERIVED from origin_class ('error'->'standard',
    -- 'complaint'->'stringent') by the machine-set classifier — not model-settable.
    -- The complaint path (fakeable text) always lands on the stricter gate.
    required_tier   TEXT NOT NULL DEFAULT 'standard' CHECK (required_tier IN ('standard','stringent')),
    -- HARD approval record (owner approval is a mechanism, not a vibe). These are
    -- set ONLY by the Phase-B /approve owner path, never by the miner or the brain:
    approved_by     TEXT,                           -- owner person_id
    approved_at     TIMESTAMPTZ,
    approval_channel TEXT,
    resolved_at     TIMESTAMPTZ,
    -- Belt-and-braces provenance invariant (cross-review). The whole security thesis
    -- is "provenance is machine-set and UN-FORGEABLE at the DATA layer" — yet the
    -- per-column CHECKs above only validate each column in isolation. Bind the tier and
    -- signal_kind to their class at the storage layer so a divergent row (e.g. a forged
    -- 'complaint' smuggled onto the lax 'standard' gate) is REJECTED by Postgres, not
    -- merely by Python convention (GapSignal.required_tier + the hardcoded detectors).
    -- This matches EXACTLY the two legitimate write paths the miner emits — error =>
    -- standard from {degraded,tool_error}; complaint => stringent from reply_phrase
    -- (capability_requests.py REQUIRED_TIER_BY_ORIGIN + _error_signals/_complaint_signals)
    -- — and holds for the row's whole lifetime (Phase B mutates only status/diagnosis/
    -- approval columns, never class/tier/kind). Same doctrine as the read_only connection
    -- and the signal_kind CHECK: convention AND mechanism, not either alone.
    CONSTRAINT v2_caprequests_provenance_ck CHECK (
        (origin_class = 'error'
             AND required_tier = 'standard'
             AND signal_kind IN ('degraded','tool_error'))
     OR (origin_class = 'complaint'
             AND required_tier = 'stringent'
             AND signal_kind = 'reply_phrase')
    )
);

-- --------------------------------------------- v2_capability_request_examples
-- The dedup + recurrence child: one row per (fingerprint, turn) that ever
-- contributed to a gap. The composite PRIMARY KEY makes counting a turn twice
-- impossible (crash-retry safe — the miner INSERTs ON CONFLICT DO NOTHING), and
-- the parent's how_often is COUNT(*) over this child so recurrence spans distinct
-- turns, never a blind increment. The PK's leading `fingerprint` column also
-- serves the count-by-fingerprint subquery, so no extra index is needed here.
-- APPEND-ONLY / UNBOUNDED by construction (cross-review): a persistently-failing
-- subsystem adds one child row every turn it fails, forever, and how_often is a
-- COUNT(*) over this child on every upsert. Volumes are tiny for Postgres near-term,
-- but this is the one place worth a future retention policy — Phase-B task: cap the
-- examples kept per fingerprint (keep the most recent N used for recurrence) or age
-- rows out beyond the recurrence window, and consider materializing how_often on write.
CREATE TABLE IF NOT EXISTS v2_capability_request_examples (
    fingerprint     TEXT NOT NULL,
    turn_id         BIGINT NOT NULL,                -- v2_turns.id that carried the signal
    seen_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fingerprint, turn_id)
);

-- Query pattern (surfacing + the /gaps read): filter/sort open gaps by recurrence
-- then recency — `WHERE status = 'open' ... ORDER BY how_often DESC, last_seen_at DESC`.
CREATE INDEX IF NOT EXISTS v2_caprequests_status_idx
    ON v2_capability_requests (status, how_often DESC, last_seen_at DESC);

-- Watermark: the miner reuses the generic high-water-mark table from migration 002
-- (v2_extraction_watermark) with source = 'capability_gaps' — one forward-only mark,
-- the raw Postgres timestamptz string stored verbatim (the ms-vs-µs precision trap).
-- No new table needed; 002 < 004 so the table already exists by the time this runs.
