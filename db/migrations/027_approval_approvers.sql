-- NULL preserves the legacy owner-only approval and posting path.
ALTER TABLE v2_approvals ADD COLUMN IF NOT EXISTS approvers text[];
CREATE TABLE IF NOT EXISTS v2_approval_posts (
    approval_id bigint NOT NULL REFERENCES v2_approvals(id),
    approver text NOT NULL,
    channel_id text,
    message_id text NOT NULL,
    shown text,
    PRIMARY KEY (approval_id, approver)
);
