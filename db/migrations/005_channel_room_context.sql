-- 005: cross-surface room context — channel_id + display_name on the turns audit.
-- Runs against the aerys_v2 database (NAS Postgres), same as migration 001's v2_turns.
--
-- WHY: the track/memory-continuity feature person-keys the checkpointer thread
-- ('person:{person_id}' for every discord/telegram surface), so thread_id NO LONGER
-- encodes WHICH shared room a turn happened in — the channel snowflake is gone from
-- the key. The channel-recent room-awareness block (last N turns of a public channel,
-- all people, injected into the system prompt) needs that room key back as a
-- first-class column, plus a speaker label to render the recent activity readably.
--
-- Both columns are NULLABLE and this is append-only: existing rows and every existing
-- reader (extraction.V2_TURNS_SQL, capability_requests.MINER_SQL — they name columns
-- explicitly) are untouched. The audit recorder (factory.turn_recorder_for) is
-- FAIL-OPEN, so a brain that hasn't yet run this migration just logs the insert
-- failure and keeps serving — a missing migration costs audit rows, never a live turn.
ALTER TABLE v2_turns ADD COLUMN IF NOT EXISTS channel_id   TEXT;   -- raw platform room id (discord channel / telegram chat)
ALTER TABLE v2_turns ADD COLUMN IF NOT EXISTS display_name TEXT;   -- speaker label for the room block ("Chris: ...")

-- The room query filters (channel_id, channel) — channel disambiguates a theoretical
-- discord-snowflake / telegram-chat-id numeric collision — and orders by created_at
-- DESC to take the most recent N. This index serves that exact shape.
CREATE INDEX IF NOT EXISTS v2_turns_room_idx ON v2_turns (channel_id, channel, created_at);
