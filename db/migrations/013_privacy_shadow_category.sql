-- 013: the privacy shadow also records Jev's sensitive-category answer (Chris approved 2026-09-26).
-- Runs against the aerys_v2 database (NAS Postgres), same as 012.
--
-- WHY: offline on 92 hand-labelled turns, the p(public) noul over-hid 17 turns at its 0.9 bar;
-- a category Choice caught the same private turns and over-hid 2. It rides the same Jev call.
-- jev_category is one word from a fixed list (ordinary/health/money/wellbeing/relationship/
-- work_confidential/secret) — still no content in this table.
ALTER TABLE v2_privacy_shadow ADD COLUMN IF NOT EXISTS jev_category TEXT;
ALTER TABLE v2_privacy_shadow ADD COLUMN IF NOT EXISTS jev_p_ordinary REAL;
