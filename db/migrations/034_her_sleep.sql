-- 034: her sleep (D100, Chris 11/25 12:48: "yes to both Tidy and Reflect"). Additive, IF NOT EXISTS.
--
-- v2_memory_proposals: what the nightly Tidy pass found. It only PROPOSES: nothing in her memory changes
-- until Chris approves a row on the dashboard's Memory tab and the apply step (python -m aerys_v2.workers
-- sleep-apply) re-checks it and retires the rows it names (memories: soft delete; core_claim: status
-- 'retired', the 9/21 mechanism). proposal_key is stable (kind + the rows + their text), so a proposal he
-- rejected is never proposed again, and a pair judged fine ('clear') is never judged again until its
-- text changes. snapshot holds each row's text as it was judged; the apply step refuses a row that changed.
CREATE TABLE IF NOT EXISTS v2_memory_proposals (
    id           BIGSERIAL PRIMARY KEY,
    proposal_key TEXT NOT NULL UNIQUE,
    person_id    UUID NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('duplicate', 'contradiction', 'stale_plan', 'old_claim')),
    retire_table TEXT NOT NULL CHECK (retire_table IN ('memories', 'core_claim')),
    retire_ids   TEXT[] NOT NULL DEFAULT '{}',
    keep_ids     TEXT[] NOT NULL DEFAULT '{}',
    snapshot     JSONB NOT NULL DEFAULT '{}'::jsonb,
    reason       TEXT NOT NULL DEFAULT '',
    score        REAL,
    status       TEXT NOT NULL DEFAULT 'proposed'
                 CHECK (status IN ('proposed', 'approved', 'rejected', 'applied', 'stale', 'clear')),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at   TIMESTAMPTZ,
    applied_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS v2_memory_proposals_status ON v2_memory_proposals (status, person_id);

-- v2_reflections: her nightly note in her own words, one per person per day. It feeds nothing else until
-- Chris has read the first week (D100): it is not a memory, so recall, search and the KNOWN list never see
-- it. pattern: the "Pattern:" line she ended with, if any (also filed to the gaps board for Kael).
CREATE TABLE IF NOT EXISTS v2_reflections (
    day        DATE NOT NULL,
    person_id  UUID NOT NULL,
    note       TEXT NOT NULL,
    pattern    TEXT NOT NULL DEFAULT '',
    turns      INTEGER NOT NULL DEFAULT 0,
    model      TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (day, person_id)
);
