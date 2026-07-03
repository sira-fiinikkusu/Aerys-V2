# LEARNING — router, home_control, and the action subgraph (the TOOLS block)

*2026-07-03. Option C, ratified: the Brain gets its first WRITE capability. Every concept
below is mapped to something you already run in n8n.*

## The one-sentence version

`router.py` spends one Haiku call deciding "does this turn need to TOUCH something?" and
drafts the acknowledgment for free, `tools/home_control.py` is the first native write
(canary-gated, outbox-audited, honest on failure), and `ask()` races the router against
chat generation so a voice device command speaks in ~800ms instead of waiting ~2s+ for
the full reply.

## router.py — the Classify Intent node reborn, with a bonus

| aerys-v2 | n8n equivalent | why it's better here |
|---|---|---|
| `RouteDecision {route, ack}` | Classify Intent → Switch node | one call yields BOTH the verdict and the speakable ack — the classifier already read the message, so drafting the ack costs nothing extra |
| `parse_route_reply` first-`{` to last-`}` slice | Parse Classification Code node | tolerates code fences (models at temp 0 still wrap JSON) but validates the OBJECT strictly — route must be exactly `"chat"` or `"action"`, nothing coerced |
| `fallback_decision` keyword heuristic | the `'haiku'`-not-in-list safety check | a dead router degrades to keywords instead of taking the turn down |
| `build_router(model, soul)` / `router_for(settings, soul)` | — | injectable model = offline tests with fakes; prod gets Haiku at temp 0, 200 max tokens, 10s timeout |

**The acks are GENERATED, never canned — owner requirement.** A templated "On it!"
heard five times a day reads as a phone tree, not a companion. The router writes the ack
fresh per message, in Aerys's voice, referencing what was actually asked ("[softly]
Dousing the office light now"). The ONLY canned string is `FALLBACK_ACK`, and it fires
exclusively on the degraded path — when the router itself is down or speaking garbage,
there is no generated ack to use.

**Failure direction is locked toward action.** A device command misrouted to chat
gaslights the caller — "done!" while the light stays off, the exact V1
hallucinated-tool-call failure mode. The action subgraph is the audited path (outbox
rows, canary gate, honest errors), so a misroute THERE is visible and safe. The keyword
heuristic is deliberately trigger-happy for the same reason.

## Parallel-start — why voice actions FEEL faster

Non-voice threads run the router sequentially (nobody's waiting on a speaker; only the
chosen path spends tokens). Voice threads (`thread_id` starts with `voice`) race:

1. Router (~300ms of Haiku) and chat generation (seconds of the daily driver) both
   launch NOW, in a two-worker pool.
2. Router says **chat** → the generation was already in flight; the router's latency
   vanished entirely inside the chat call's shadow. Zero cost to the common case.
3. Router says **action** → the caller gets the generated ack IMMEDIATELY (~800ms
   end-to-end vs ~2s+ for a full reply — the ~3.6s voice budget stays intact) while a
   background thread runs the tool loop and appends the REAL outcome to the SAME thread
   via `update_state(as_node="chat")` — so next turn, the model's history shows what
   actually happened, not just the ack.

The transport contract never changes: `http_api.py` still does one-request-one-string.
HA speaks whatever comes back; the asynchrony lives entirely behind the `ask()` seam. In
n8n this trick was impossible — a workflow can't return early AND keep executing into
the same conversation history.

## home_control — the first native write, three gates deep

n8n mapping: this replaces 07-01 "HA Action: Play Music (owner-gated)" as the
pattern-setter — the same `POST /api/services/<domain>/<service>` the HTTP Request node
hit, minus the workflow around it. The function is literally named `home_control`
because of the V1 toolWorkflow lesson: the name the LLM sees MUST match what prompts
call it, or the model hallucinates having called it.

| Gate | What it refuses | n8n ancestor |
|---|---|---|
| domain gate | anything outside `light`/`switch` — misfires are annoyances, not hazards | the owner-gate on 07-01 |
| **canary allowlist** | writes to any entity not in `HA_CANARY_ENTITIES`; reads stay unrestricted (looking can't break anything) | crawl-walk-run, now enforced in code instead of convention |
| honest failure | nothing — it refuses to LIE: HA unreachable / 4xx / refused all return plain error strings the model must relay; never an exception (an exception inside ToolNode kills the whole action turn — the failed-webhook-kills-execution outage mode, again) | the "never claim success" half lives in `ACTION_OVERLAY` prompt-side |

The canary allowlist is what bounds the blast radius: the beta can only touch the
entities you explicitly listed, and when the tool refuses, the model tells the caller
the truth ("the only entities I may control are: …") instead of pretending.

## Outbox-inline, and the lease exception — documented honestly

Every write that reaches HA is write-ahead audited: INSERT the intent into `v2_outbox`
as `'executing'` → call HA → UPDATE with receipt (HA's changed-states list = evidence,
not a bare ok) or failure. Two separate short connections on purpose — a crash mid-call
leaves exactly the `'executing'` row the sweeper contract expects, never a silent
mystery write. Audit trouble never costs the turn, but it logs loudly: an unaudited
write is a real event.

**The one-armed-writer exception (deliberate, bounded, beta only).** The
`v2_writer_lease` rule says a write capability REFUSES unless it holds the lease for its
kind — and `ha_write`'s lease still says `'n8n'`. But the callers reaching this tool are
satellite-scoped voice threads n8n never serves, so a double-fire is structurally
impossible for this slice. Owner-ratified: execute anyway, and mark every such payload
`{"lease_exception": "beta-canary"}` — queryable
(`SELECT .. WHERE payload ? 'lease_exception'`), and the exception dies loudly when the
lease flips to `'brain'` and the marker stops appearing. This is the ONLY capability
allowed to bend the rule, and the bend is in the audit trail, not around it.

## The tests — 30 new, all offline

`test_router.py` pins the parse contract (fences tolerated, bad routes rejected, dead
model → heuristic), `test_home_control.py` runs the tool against an
`httpx.MockTransport` + fake conn_factory (canary refusals, outbox open/close, lease
marker), and `test_action_orchestration.py` proves the seams: backward-compatible
without a router, non-voice sequential routing, voice ack-then-act, and honest failure
landing in the thread. 298 green total, no API key, CI-safe.

## Try it yourself

```bash
uv run pytest -q                                        # 298 green
HA_TOKEN=... HA_CANARY_ENTITIES=light.office_lamp \
  uv run aerys-v2 --serve                               # arms the TOOLS block
# then, over voice: "turn on the office light" → ack in ~800ms, light follows
```

## What's deliberately NOT here yet

Locks, covers, climate (those arrive with confirmation semantics, not just an
allowlist), the outbox sweeper that reconciles stuck `'executing'` rows, flipping the
`ha_write` lease to `'brain'` (retires the exception), streaming, and any second tool.
One write capability, gated three ways, before the toolbox grows.

## Field notes from the first live week (2026-07-03)

**Per-satellite follow-up routing is a known limitation.** `HA_ANNOUNCE_ENTITY` is a
single static target — the OpenAI shim never receives satellite identity from HA
(Extended OpenAI Conversation's chat-completions payload carries no device/satellite
field), so "announce where the request came from" cannot be implemented at this seam.
For the single-satellite beta, point the env var at the satellite actually running the
pipeline (bit us live: it pointed at the Voice PE while the beta pipeline lived on the
ReSpeaker, so follow-ups spoke in the wrong room). The proper fix — satellite identity
riding the request — belongs to the voice-runtime phase.

**STT garble vs the one-way announce channel.** Live incident: "turn office light one
off" transcribed as "Can you turn off office light on?" — the router acked "off", then
the action subgraph (correctly reading an ambiguous sentence) ASKED "did you mean on or
off?" over a channel the user cannot answer, contradicting the ack already spoken.
Fix: the router's ack now rides `configurable.spoken_ack` into the action subgraph,
whose prompt (VOICE_ACK_OVERLAY) says: never ask on this path; resolve garble toward
the acknowledged reading; if truly unexecutable, state the problem in one sentence.
Phoenix trace evidence also confirmed the subgraph's input really is just the current
turn (no thread-history contamination) — the ambiguity came in through the STT text
itself, not through leaked ping-pong history.
