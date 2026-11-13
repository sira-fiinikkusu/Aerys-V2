-- 021's inline status CHECK is named v2_home_notices_status_check by PostgreSQL.
-- Drop/add (applied inside the deploy's one transaction), so a repeat keeps the same constraint.
ALTER TABLE v2_home_notices
    DROP CONSTRAINT IF EXISTS v2_home_notices_status_check;
ALTER TABLE v2_home_notices
    ADD CONSTRAINT v2_home_notices_status_check
    CHECK (status IN ('success', 'error', 'cancelled', 'interrupted', 'timeout', 'check_in'));
