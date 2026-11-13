-- 032: a message he asks for when a wake-up is set (D42, 2026-11-13, board #194). His ask is the
-- approval, only on a turn where sending right then would have needed no button (his own text or his
-- voice surely his, nothing outside read). The wake-up keeps who (resolved then: person id, Discord
-- address, name, handle) and the exact words; at its time a send of THOSE words to THAT person goes
-- once per run without his tap. Anything else a wake turn sends still waits for his tap.
--   pinned_send  written at add, never updated afterwards (wakeups.py refuses it, like the note)
--   pinned_run   the run that spent it; spend_pin sets it atomically before the send goes
-- Idempotent: safe to run twice.
ALTER TABLE v2_wakeups ADD COLUMN IF NOT EXISTS pinned_send JSONB;
ALTER TABLE v2_wakeups ADD COLUMN IF NOT EXISTS pinned_run INTEGER NOT NULL DEFAULT 0;
