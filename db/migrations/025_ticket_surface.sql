-- 025: where a ticket was asked, and where a text-only reach goes (board #176, D20; the owner
-- 2026-11-09: "if i ask her in discord she should respond to tickets from discord").
-- asked_on: the surface of the turn that opened the ticket (discord | telegram | voice | lens | kael |
-- NULL for anything else and for tickets older than this). text_to: the text surface a text_only
-- reach is delivered on (NULL = the desk's default text surface, as before).
ALTER TABLE v2_kael_tickets ADD COLUMN IF NOT EXISTS asked_on TEXT;
ALTER TABLE v2_reachouts ADD COLUMN IF NOT EXISTS text_to TEXT;
