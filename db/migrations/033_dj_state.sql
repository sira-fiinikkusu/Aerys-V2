-- D52: shared DJ controls and activity across serve/Discord/Telegram processes.
-- Fixed control/break keys plus one voice/command/previous key per configured device/player.
CREATE TABLE IF NOT EXISTS v2_dj_state (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL
);
