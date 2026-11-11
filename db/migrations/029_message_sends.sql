-- Durable reservation before network delivery; no words or handles in this ledger.
CREATE TABLE IF NOT EXISTS v2_message_sends (
    id bigserial PRIMARY KEY,
    attempted_at timestamptz NOT NULL DEFAULT now(),
    person_id text NOT NULL,
    platform text NOT NULL CHECK (platform = 'discord'),
    words_hash text NOT NULL,
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'ok', 'failed'))
);
CREATE INDEX IF NOT EXISTS v2_message_sends_recent
    ON v2_message_sends (person_id, platform, words_hash, attempted_at DESC);
