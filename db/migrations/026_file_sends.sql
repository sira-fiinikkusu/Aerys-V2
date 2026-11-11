-- 026: a record of every file she hands the owner (board #187, 11/10).
-- On 11/10 23:13 he asked by voice "what file did you send me tonight?". Her hands looked in the
-- home-job records, found no send (the 22:14 send was a direct send_file_to_chris, never a job),
-- decided she hadn't sent it, and sent it again. Her own words were her only evidence of a send.
-- send_file_to_chris writes a row here once Discord or Telegram took the file; files_sent_to_chris
-- reads it back, and the same file to the same person within hours needs him to ask again.
-- Additive and idempotent: a new table, nothing existing changes.
CREATE TABLE IF NOT EXISTS v2_file_sends (
    id         BIGSERIAL PRIMARY KEY,
    sent_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    person_id  TEXT NOT NULL,
    path       TEXT NOT NULL,              -- relative to her home workspace
    name       TEXT NOT NULL,
    bytes      INTEGER NOT NULL CHECK (bytes >= 0),
    surface    TEXT NOT NULL CHECK (surface IN ('discord', 'telegram')),
    thread_id  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS v2_file_sends_person_path ON v2_file_sends (person_id, path, sent_at DESC);
CREATE INDEX IF NOT EXISTS v2_file_sends_person_time ON v2_file_sends (person_id, sent_at DESC);
