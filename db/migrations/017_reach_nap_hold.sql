-- 017: a nap holds her like a meeting does (board #176, Stage 2). The owner, 2026-11-08,
-- picked it: when he tells her he's napping, nothing she'd say aloud goes until he talks to
-- her again or an hour passes; timely things come as silent text. hold_since is when he
-- said it; hold_until is the latest it lasts.
ALTER TABLE v2_reach_prefs ADD COLUMN IF NOT EXISTS hold_since TIMESTAMPTZ;
ALTER TABLE v2_reach_prefs ADD COLUMN IF NOT EXISTS hold_until TIMESTAMPTZ;

-- When each turn STARTED. created_at is when the audit row lands, which can trail the end
-- of the turn (the write runs off the hot path), so "did he talk to her after he said he
-- was napping?" reads the start instead. NULL on rows from before this migration.
ALTER TABLE v2_turns ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;
