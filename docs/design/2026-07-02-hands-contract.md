# Design: the Brain↔Hands contract (1j)

*2026-07-02. Status: DESIGN — the first Hand wires up when the first write capability
migrates. This is the coexistence-era API between the Python Brain and the n8n
workflows that keep doing jobs the Brain hasn't absorbed yet.*

## Scope — which Hands survive short-term

From the migration sequencing: **DB-writing capabilities** (memory writes, governance
writes, profile overrides) and **email send** (dead last by design). Everything
read-only is already native Python (services/). A Hand is a *temporary* arrangement:
each one carries retirement criteria, not a permanent seat.

## The contract shape

One production n8n webhook per capability, versioned by path:

```
POST http://jetson.local:5678/webhook/hands/<capability>-v1
Headers: X-Hands-Key: <shared secret, .env both sides>
Body:    { "intent_id": "<v2_outbox idempotency_key>",
           "person_id": "<uuid>",
           "payload": { ...capability-specific... } }
Reply:   { "status": "ok" | "duplicate" | "error",
           "receipt": { ...capability-specific evidence... },
           "error": "<present when status=error>" }
```

Rules, in order of load-bearing-ness:

1. **Every call rides the outbox.** The Brain never calls a Hand directly from a graph
   node: it inserts the `v2_outbox` row (write-ahead), and the outbox executor makes
   the HTTP call, records the receipt, marks the row. Crash anywhere = reconcilable
   pending row. This is the same lifecycle as native side effects — a Hand is just an
   executor whose implementation happens to be n8n.
2. **The Hand dedupes on `intent_id`.** n8n side keeps a tiny `hands_processed` table
   (intent_id PK, receipt, processed_at). First action per intent executes; replays
   return `duplicate` + the original receipt. This makes Brain-side retries safe and
   is the n8n mirror of the outbox's idempotency key.
3. **The lease still rules.** The outbox executor checks `v2_writer_lease` before
   POSTing. When a capability's lease flips from `n8n` to `brain`, the Hand webhook
   stays up but stops receiving — retire it only after the retirement criteria below.
4. **Success needs evidence.** A Hand returns a receipt with verifiable content (row
   id, message id, gmail id) — never a bare `{"ok": true}`. The n8n Postgres
   UPDATE-collapse quirk makes "it said success" worthless; receipts are the antidote,
   and the outbox stores them.
5. **Sync reply, hard timeout.** Hands respond synchronously (Respond to Webhook,
   typeVersion 1 — the known-good config). Brain-side timeout 30s; a timeout is NOT a
   failure verdict — it's `status unknown`, and the executor re-polls by intent_id
   (rule 2 makes that free) before retrying.

## Why webhooks and not the n8n API

Executing workflows via the public API isn't available on this instance, and temp
webhooks are broken — but *production* webhooks registered at activation are proven
(kael-dm has run for months). Hands are production workflows with stable paths.

## Retirement criteria (per Hand)

A Hand retires when: (a) the native implementation has run its canary period with the
lease, (b) `hands_processed` shows zero non-duplicate calls for 14 days, and (c) the
workflow export is committed to the deploy repo. Then deactivate — the webhook 404s,
and any straggler call fails loudly at the executor (alerted), not silently.

## Deferred on purpose

The n8n-side Hand template workflow (build with the first migrating write, not
before); HMAC-signed bodies (shared secret suffices on LAN; revisit if a Hand is ever
exposed past the tunnel); capability schema registry (one dataclass per Hand payload
in `contract/` when the first Hand lands).
