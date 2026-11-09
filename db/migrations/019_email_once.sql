-- 019: one ledger row per email, even with two watchers at once (board #176, Stage 2). The
-- email watcher (its own container) writes its pings into v2_reachouts for the brain's desk to
-- deliver, keyed mailbox:uidvalidity:uid; it inserts only when no row has that key (in any
-- state, ever). This index makes that hold when two producers race: the loser's insert does
-- nothing. Scoped to the watcher's rows, so the desk's own dedupe (a 24-hour window) is unchanged.
CREATE UNIQUE INDEX IF NOT EXISTS uq_reachouts_email_key ON v2_reachouts (dedupe_key)
    WHERE source = 'email_watch';
