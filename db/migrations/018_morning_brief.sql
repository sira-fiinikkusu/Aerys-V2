-- 018: her morning brief (board #176, Stage 2). The owner, 2026-11-08: Kael's brief stays on
-- his phone and watch; hers is spoken, "in the office when she first detects me in there after
-- waking", so he's more awake and it doesn't wake his wife. The brief goes through the desk like
-- everything else she starts, with three things only a brief needs:
--   room   the one room it may be spoken in (it waits for him there, never follows him)
--   brief  a brief: outside the daily budget (D4 "3 a day plus briefs"), and the night doesn't
--          hold it, because it only exists once he's up and in that room
--   folds  the routine items held overnight that it carries; when it's delivered they're
--          marked 'folded' instead of going out one by one
ALTER TABLE v2_reachouts ADD COLUMN IF NOT EXISTS room TEXT;
ALTER TABLE v2_reachouts ADD COLUMN IF NOT EXISTS brief BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE v2_reachouts ADD COLUMN IF NOT EXISTS folds BIGINT[] NOT NULL DEFAULT '{}';
ALTER TABLE v2_reachouts DROP CONSTRAINT IF EXISTS v2_reachouts_status_check;
ALTER TABLE v2_reachouts ADD CONSTRAINT v2_reachouts_status_check
    CHECK (status IN ('queued', 'sent', 'dropped', 'failed', 'folded'));
