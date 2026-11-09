-- 024: his home jobs, across turns and surfaces (board #176, s3-02). Her action turns are one-shot,
-- so the official AsyncSubAgentMiddleware's task records (async_tasks) would vanish when a turn
-- ends; home_jobs.py keeps them here, keyed to him, and hands them back to the tools each turn.
-- task is the official record (task_id, agent_name, thread_id, run_id, status, timestamps); the
-- job's words never land here.
CREATE TABLE IF NOT EXISTS v2_home_tasks (
    task_id    TEXT PRIMARY KEY,
    owner      TEXT NOT NULL,
    task       JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_home_tasks_owner ON v2_home_tasks (owner, updated_at DESC);
