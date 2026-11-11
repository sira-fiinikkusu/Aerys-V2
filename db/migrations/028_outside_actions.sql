-- D27/D28: exact tool arguments held for the owner's approval after outside text.
ALTER TABLE v2_gated_actions DROP CONSTRAINT IF EXISTS v2_gated_actions_action_check;
ALTER TABLE v2_gated_actions ADD CONSTRAINT v2_gated_actions_action_check
    CHECK (action IN ('home_job', 'message', 'outside_action'));
