# Design: turns audit + action outbox — the migration spine

*2026-07-02. Status: SCHEMA COMMITTED (`db/migrations/001_turns_and_outbox.sql`), writers
land with Phase 2 (Postgres wiring). Cross-review verdict: build this pattern first —
every earlier canary otherwise invents its own incompatible confirmation/delivery
mechanism.*

## Why two tables and a lease

**Problem 1 — forensics.** The checkpointer deliberately stores messages only. But
incidents and evals need: who was resolved, which tier fired and why, what the model
actually said before any post-processing, and whether delivery happened. That's
`v2_turns` — append-only audit, one row per `ask()` turn, never read on the hot path.

n8n mapping: this replaces squinting at n8n's execution list (which self-destructs after
30 days and collapses UPDATE outputs). Every question you've ever answered by opening a
workflow execution becomes a SQL query.

**Problem 2 — side effects that can't lie.** A Discord send can fail after the
checkpoint commits, or succeed right before a crash. Without a record, a retry
duplicates the message; without a retry, it's lost. `v2_outbox` is write-ahead intent:
INSERT the intent → execute → record receipt. A crash leaves a `pending` row a sweeper
reconciles — never a duplicate, never silent loss. `idempotency_key` (unique) makes
retries collide with themselves safely.

n8n mapping: this is what the "Send Discord Message retryOnFail" band-aid wanted to be.
It's also the general form of the email draft/confirm flow — `requires_confirmation` +
`confirmation_binding` park an action until the RIGHT person says yes, and
`expires_at` means stale confirmations rot instead of firing weeks later.

**Problem 3 — exactly one armed writer.** During coexistence, n8n AND the Brain can
both technically send messages / write HA / send email. Policy sentences don't stop a
stale workflow from double-firing. `v2_writer_lease` is mechanical: every write
capability checks `holder` for its kind before firing. Cutover = one UPDATE
(`'n8n'`→`'brain'`); rollback = the same UPDATE back. The lease row IS the cutover
switch, which makes every cutover step in the dossier's §4 table a reversible one-liner.

## Lifecycle (emit example)

```
ask() turn completes
  → INSERT v2_outbox (kind='emit', payload={channel, text, chunks}, idempotency_key=turn:<id>:emit)
  → executor checks v2_writer_lease['emit'].holder == 'brain'   (else: log + drop — n8n owns it)
  → status 'executing' → Discord send → receipt {"message_id": ...} → 'succeeded'
  crash anywhere → row stays 'pending'/'executing' → sweeper retries with the SAME
  idempotency key; a second 'succeeded' insert is impossible (UNIQUE)
```

## What deliberately isn't here

- No queue infrastructure (Redis etc.) — the outbox table + a sweeper loop IS the queue,
  homelab-sized.
- No generic workflow engine — `kind` stays a short enum; if a kind needs branching
  logic, that logic lives in its executor function, not in rows.
- Turns are not checkpoints — replaying a conversation still comes from the LangGraph
  checkpointer; `v2_turns` answers "what happened," not "what was the state."

## Open items (deliberate)

- Sweeper cadence + backoff policy → with the Phase 2 worker process (scheduler lives
  out-of-process per cross-review #1).
- Retention: turns grow forever; decide archival policy when the table has real weight.
- `trace_id` joins to Phoenix once tracing lands (01-05).
