-- 011: keep the shadow judgment beside the router she actually followed.
-- Runs against the aerys_v2 database (NAS Postgres), same as 001 and 010.
--
-- WHY: the Jev design needs evidence from real turns before it can steer one.
-- Both verdicts belong on the same row, including failures, so agreement can be
-- measured without reconstructing a router call or silently losing timeouts.
--
-- Append-only and nullable: old rows and default-off turns have no shadow result.
-- The recorder is FAIL-OPEN; a missing migration loses audit rows, never replies.
ALTER TABLE v2_turns ADD COLUMN IF NOT EXISTS reflex JSONB;
