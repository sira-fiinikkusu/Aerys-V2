-- 011: remember which gaps have already been put on the shared board.
-- Runs against the aerys_v2 database, same as 001 / 005 / 007 / 010.
--
-- WHY: she files her own gaps with log_gap, and the miner files what it reads from
-- her turns. Both land in v2_capability_requests, which only the owner's /gaps command
-- ever reads. Chris, 2026-09-14: "since she has her own gh identity ... if we could
-- utilize that further for general gaps reporting in addition to what we have."
--
-- The table stays the source of truth and the trust lane is unchanged. The board is
-- where the work actually happens — where he and Kael comment, label, close and link
-- commits — so a gap that never reaches it is a gap nobody works.
--
-- This column is the "already published" marker, so publishing is idempotent and a
-- gap can never be filed twice. NULL = not yet on the board.
--
-- Append-only and nullable: every existing reader names its columns explicitly, and
-- the publisher is a separate pass that can be re-run or skipped entirely.
ALTER TABLE v2_capability_requests ADD COLUMN IF NOT EXISTS board_issue INTEGER;

CREATE INDEX IF NOT EXISTS v2_caprequests_unpublished_idx
    ON v2_capability_requests (status)
    WHERE board_issue IS NULL;
