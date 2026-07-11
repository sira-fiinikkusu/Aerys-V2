-- 006: email inbox watcher — the UID high-water mark for Aerys's OWN Gmail inbox.
-- Runs against the aerys_v2 database (NAS Postgres), same home as the checkpointer,
-- outbox, and the other v2_* worker tables. Forward only, non-destructive: IF NOT
-- EXISTS throughout, nothing DROPs or DELETEs — a re-run is a no-op.
--
-- Worker: workers/email_watch.py (notification-only — new-mail pings to the owner;
-- reading/sending mail is a separate future tool). The worker reads and writes ONLY
-- this table; it never touches prod `aerys`.

-- ------------------------------------------------------------ v2_email_watermark
-- One row per watched mailbox. The stored values are the IMAP SERVER'S OWN integers,
-- verbatim — the generalized form of extraction's ms-vs-µs watermark lesson: never
-- persist a locally re-derived cursor (a fetch timestamp, a re-serialized value)
-- when the server hands you its own monotonic one.
--
--   uidvalidity — the mailbox's UIDVALIDITY at last check. When the server changes
--     it (a Gmail reindex / mailbox recreation), every UID is renumbered and
--     last_uid becomes meaningless: the worker RESETS last_uid to the current max
--     UID and pings NOTHING — a reindex must never replay 500 "new mail" pings.
--   last_uid — the highest UID already pinged (or adopted at first run / reset).
--     Advanced PER MESSAGE as pings succeed, so a mid-burst notify failure holds
--     the mark below the un-pinged message (deferred to next tick, never lost)
--     without re-pinging the ones already delivered.
--
-- Why not reuse migration 002's v2_extraction_watermark: that table stores one raw
-- timestamptz string per source; this cursor is a (uidvalidity, uid) integer PAIR
-- with reset semantics of its own. Jamming two ints into a text column invites
-- exactly the parse-and-reformat bugs the verbatim rule exists to prevent.
CREATE TABLE IF NOT EXISTS v2_email_watermark (
    mailbox      TEXT PRIMARY KEY,                      -- e.g. 'INBOX'
    uidvalidity  BIGINT NOT NULL,                       -- server's value, verbatim
    last_uid     BIGINT NOT NULL,                       -- server's value, verbatim
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
