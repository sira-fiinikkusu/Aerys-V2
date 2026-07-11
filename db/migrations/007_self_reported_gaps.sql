-- 007: self-reported gaps — a third legitimate write path into
-- v2_capability_requests: the `log_gap` TOOL (tools/log_gap.py), where the
-- model deliberately files a gap instead of the miner inferring one from
-- reply text after the fact.
--
-- Origin story (2026-07-11): the owner asked her, by voice on the glasses, to
-- "log a gap" — she honestly had no tool for it, AND the miner's fixed
-- complaint-phrase list then missed both her paraphrase ("don't have a
-- logging tool actually wired up") and her clean "**Issue:** ..." statement.
-- One experiment, two findings; this migration + the tool close both.
--
-- Trust model: a self-reported gap is MODEL-AUTHORED text end to end — the
-- same trust class as a mined complaint, so it binds to the SAME lane:
-- origin_class='complaint', required_tier='stringent'. The new signal_kind
-- 'self_reported' records the MECHANISM (deliberate tool call vs mined
-- phrase) without opening a softer gate. The provenance CHECK is extended,
-- not relaxed: error rows still cannot carry it, and a forged 'standard'
-- self-report is still rejected by Postgres, not just by Python convention.
--
-- Forward-only, non-destructive: constraint swaps only, no data touched.
-- Re-run is a no-op in effect (DROP IF EXISTS + ADD of identical text).

ALTER TABLE v2_capability_requests
    DROP CONSTRAINT IF EXISTS v2_capability_requests_signal_kind_check;
ALTER TABLE v2_capability_requests
    ADD CONSTRAINT v2_capability_requests_signal_kind_check
    CHECK (signal_kind IN ('degraded','tool_error','reply_phrase','self_reported'));

ALTER TABLE v2_capability_requests
    DROP CONSTRAINT IF EXISTS v2_caprequests_provenance_ck;
ALTER TABLE v2_capability_requests
    ADD CONSTRAINT v2_caprequests_provenance_ck CHECK (
        (origin_class = 'error'
             AND required_tier = 'standard'
             AND signal_kind IN ('degraded','tool_error'))
     OR (origin_class = 'complaint'
             AND required_tier = 'stringent'
             AND signal_kind IN ('reply_phrase','self_reported'))
    );
