# LEARNING — the migration spine: v2_turns, v2_outbox, writer lease, AlertSink

*2026-07-02. Companion to `docs/design/2026-07-02-turns-outbox-spine.md`. The schema is
committed (`db/migrations/001_turns_and_outbox.sql`); the code half is `src/aerys_v2/alerts.py`.
Writers land in Phase 2 — this doc is about understanding the shapes before anything
writes through them.*

## The one-sentence version

Every V1 reliability lesson — vanishing execution logs, retry band-aids, the IPC
watchdog dance, the error handler that only covered 19 of 33 workflows, and today's
"the webhook said 200 but nothing arrived" backup incident — becomes four small,
boring, queryable things: an audit table, an intent table, a lease row, and a sink class.

## v2_turns — the execution list you actually own

One row per `ask()` turn. Append-only, never read on the hot path.

| Question you ask today | How you answer it in n8n | How you answer it with v2_turns |
|---|---|---|
| "What did the model say before the polisher touched it?" | You can't — the execution shows the final payload; intermediate node data expires | `SELECT raw_reply, emitted_reply FROM v2_turns WHERE ...` — both copies, forever |
| "Why did this route to opus?" | Open the execution, find Parse Classification, read its output — if it's under 30 days old | `classifier_intent`, `tier`, `tier_override_source` columns on the row |
| "Who did the resolver think this was?" | Dig through Identity Resolver's execution (separate workflow, separate execution ID) | `person_id`, `platform_identity`, `resolver_version` — snapshotted on the same row |
| "Did anything degrade this turn?" | You find out when someone complains | `degraded` JSONB, e.g. `["ha_unreachable"]` |
| "Was voice fast enough this week?" | You don't | `latency_ms`, aggregatable |

The n8n execution list self-destructs after 30 days, collapses UPDATE outputs
(CLAUDE.md quirk), and can only be read by clicking through a UI. `v2_turns` is that
same forensic record as SQL — every question you've ever answered by opening an
execution becomes a query.

**Why it's a separate table from the checkpointer:** the checkpointer stores *messages*
(what the conversation was); turns store *decisions and outcomes* (what the system did).
Replaying a conversation uses the checkpointer. Investigating an incident uses turns.
Mixing them is how V1's session-contamination class of bug happens.

## v2_outbox — what retryOnFail wanted to be

The V1 pattern: Send Discord Message occasionally failed on DNS, so we bolted
`retryOnFail: true, maxTries: 3` onto the node. That handles *transient* failure on
*that one node* — and nothing else. It can still double-send (retry after a send that
actually landed), still silently lose (all 3 tries fail, execution ends, nobody knows),
and had to be copy-pasted onto every sending node in every workflow.

The outbox inverts it: record the intent *before* acting, record the receipt *after*.

```
INSERT intent (status='pending', idempotency_key='turn:42:emit')
  → executor picks it up → status 'executing'
  → Discord send → receipt {"message_id": "..."} → status 'succeeded'
```

| Failure | retryOnFail band-aid | outbox |
|---|---|---|
| Crash after send, before recording | retries → **duplicate message** | sweeper retries with the SAME `idempotency_key`; UNIQUE constraint makes a second success impossible |
| All retries exhausted | execution fails, message **silently lost** | row sits at `failed` with `last_error` — visible, sweepable, alertable |
| Owner-gated action needs a yes first | doesn't exist — every workflow invents its own confirm flow | `requires_confirmation` + `confirmation_binding` (WHO must say yes) + `expires_at` (stale confirmations rot, they don't fire weeks later) |

One table covers message sends, HA writes, governance writes, email — every side
effect, one pattern, instead of a per-workflow improvisation.

## v2_writer_lease — the IPC watchdog liturgy, replaced by one row

You know the liturgy by heart: deactivate DM adapter, deactivate guild adapter, sleep 3,
activate DM first, sleep 8, activate guild LAST — because katerlol IPC is
last-one-activated-wins and getting the order wrong leaves an adapter dead. That whole
ritual exists because "exactly one listener" was *emergent behavior* nobody could see
or set directly.

The lease makes it a **stated fact in a table**:

```sql
SELECT holder FROM v2_writer_lease WHERE kind = 'emit';   -- 'n8n' or 'brain'
```

Every write capability checks its kind's lease before firing and refuses if it's not
the holder. During migration coexistence, n8n and the Brain can both *technically* send
messages — the lease is what mechanically stops a stale workflow from double-firing.

- Cutover = `UPDATE v2_writer_lease SET holder='brain' WHERE kind='emit';`
- Rollback = the same UPDATE back to `'n8n'`.
- No sleep 3, no sleep 8, no ordering, no watchdog service. One row per capability.

The migration ships with all four kinds (`emit`, `ha_write`, `governance`, `email`)
seeded to `'n8n'` — the Brain starts disarmed by default.

## AlertSink (alerts.py) — Central Error Handler, minus the two failure modes

V1's Central Error Handler (rpxcFfLtyjhSG2Qx) was the right idea with two structural
holes:

1. **Coverage was opt-in.** It had to be attached workflow-by-workflow, and it was
   attached to 19 of 33. An error in the other 14 vanished. `AlertSink` is a plain
   object passed to whatever needs it — anything wired through `ask()` or the outbox
   executors gets covered because it's the same instance, not a per-workflow checkbox.
2. **The alerter could be the outage.** If the error-handler workflow itself broke,
   errors went nowhere and told no one. The sink's contract is *any failure inside the
   sink degrades to a log line* — `alert()` never raises, ever. Alerting about an error
   must not become the error.

| Central Error Handler | AlertSink |
|---|---|
| n8n error-trigger workflow, attached to 19/33 workflows | one class, covers everything constructed with it |
| posts to #echoes via Discord node | POSTs to the existing kael-dm webhook (same one the backup pipeline uses — zero new infra, stdlib `urllib`, zero new dependencies) |
| its own failure = silence | its own failure = `log.error`, and the return value says so |
| trusts the HTTP 200 | **checks `message_id` in the response body** |

**Why `message_id: null` counts as failure — today's lesson.** The kael-dm webhook can
return HTTP 200 with `message_id: null` when the message doesn't actually deliver
(oversize bodies do this — hence the `text[:1900]` cap with headroom under Discord's
2000). Today's backup incident was exactly this: the script believed its 200 and the
alert never existed. In n8n terms this is the presence-gate placeholder trap wearing a
different hat — a success *envelope* around a failed *delivery*. So `alert()` returns
`True` only when `message_id` is truthy; a 200-with-null logs
`"accepted but message_id null — not delivered"` and returns `False`. Never believe an
alert landed when it didn't.

## Try it yourself

```bash
# read the schema top to bottom — it's 80 lines and every column has a comment
less ~/projects/aerys-v2/db/migrations/001_turns_and_outbox.sql

# the sink in log-only mode (no webhook_url = safe anywhere, nothing sends)
cd ~/projects/aerys-v2
uv run python -c "
from aerys_v2.alerts import AlertSink
import logging; logging.basicConfig(level=logging.INFO)
sink = AlertSink(None)
print('delivered:', sink.alert('hello from the learning doc'))"
# → logs the alert, prints 'delivered: False' — logged, not sent, didn't raise

uv run pytest tests/test_alerts.py -q   # the never-raises contract, proven
```

## What's deliberately NOT here yet

- **No writers.** Nothing inserts into v2_turns/v2_outbox yet — that lands with Phase 2
  Postgres wiring. The schema ships first so earlier canaries can't invent incompatible
  local versions of confirmation/delivery (the cross-review's whole point).
- **No sweeper.** The retry loop that reconciles stuck `pending`/`executing` rows comes
  with the Phase 2 worker process (scheduler lives out-of-process, cross-review #1).
- **No queue infra.** The outbox table + sweeper IS the queue — homelab-sized on purpose.
- **No retention policy.** Turns grow forever until the table has real weight; decided then.
- `trace_id` joins to Phoenix when tracing lands (01-05).
