-- 009: storm-watch durable dedup (owner incident 2026-08-27: six brain
-- redeploys in one day re-announced Tropical Storm Dolly on every boot,
-- because the watcher's seen-state was deliberately in-memory — "a redeploy
-- at worst repeats one nudge" was written for rare deploys, not deploy-heavy
-- build days). One row per thing already announced; the watcher loads this at
-- startup and a reboot stops being amnesia. Alerts are pruned as they expire
-- (mirroring the in-memory set), storms and the outlook high-water persist
-- for the season.
CREATE TABLE IF NOT EXISTS v2_storm_seen (
    kind    TEXT NOT NULL,   -- 'alert' | 'storm' | 'outlook_hw'
    key     TEXT NOT NULL,   -- alert id / storm id / 'high_water'
    value   TEXT,            -- outlook_hw carries the number here
    seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (kind, key)
);
