# LEARNING — build_context: long-term memory crosses the thread boundary

*2026-07-03. Doc 05 built the read services; this wires them into every turn. On 7/3,
live on voice, she said a version of "that won't survive this session" — a caveat that
was TRUE when the soul was written and is now structurally false. This doc is about
making it false correctly: capability claims that follow facts, and a memory pipeline
that can never take a turn down with it.*

## The one-sentence version

`build_context()` folds profile + scored memories into one prompt-ready block,
`context_fn_for()` injects it into the chat node behind a read-only connection to the
prod aerys database, and the system prompt only *claims* cross-conversation recall
when retrieval is actually wired — so what she says about her memory is now derived
from what her memory can actually do.

## The story: the caveat dissolves

The soul was written for a brain that couldn't see its own memory, so she hedged —
"I won't remember this next time" — even after the Postgres checkpointer made threads
durable. That's the anti-UNDERclaim problem (the mirror image of the presence-gate
placeholder trap: there, models claimed abilities they lacked; here, she disclaimed
one she has). Two moves fix it:

1. **A capability overlay in the chat node** tells her the thread is durable. Always —
   the checkpointer is unconditional.
2. **A second sentence — "you also know long-term facts about the caller, across ALL
   conversations and channels" — exists ONLY when `context_fn is not None`.** Claims
   follow facts: if `MEMORIES_DATABASE_URL` isn't set, the promise is never made.
   `test_no_context_fn_means_no_header_and_no_recall_claim` pins it both ways.

## The big mapping

| V1 (n8n) | aerys-v2 | what changed |
|---|---|---|
| Core Agent prompt-builder Code node calling TWO webhooks per message (04-03 Profile API, 04-02 Memory Retrieval) and splicing the responses | `build_context(person_id, query_text, conn, embed=...)` — two function calls and a string join | no HTTP hop; a failure is a caught exception, not a dead execution |
| "Generate Embedding" HTTP Request node (which **wipes all item JSON** — hence the Pre-Embed Context recovery node) | `embedder_from_settings(settings)` — stdlib urllib, any OpenAI-compatible `/embeddings` host | surrounding variables just stay in scope |
| `n8n_chat_histories` session_id = person_id, identity resolved per-adapter | HTTP/voice callers resolve to `owner_person_id`; the chat node passes `identity["user_id"]` + the latest human turn | voice-Chris retrieves HIS memories, not an anonymous "http-caller" bucket |

## The GRACEFUL contract — the load-bearing part

In V1 a failed webhook killed the whole execution: **Aerys went mute because a SELECT
hiccupped.** That failure class is deleted here, with three independent fences:

1. **Inside `build_context`** — profile and memories are each wrapped in their own
   try/except. Profile trouble must not cost the memories, and vice versa
   (`ExplodingConn` in the tests proves a raising profile query still yields memories).
2. **Inside `context_fn_for`'s closure** — the `psycopg.connect` itself is fenced. A
   NAS outage or DNS hiccup = empty context, never a dead turn.
3. **Inside the chat node** — even if the seam breaks its promise, a raising
   `context_fn` is caught there too (`test_raising_context_fn_never_kills_the_turn`:
   the reply is still "still alive").

The worst case at every layer is *less context*, never a lost turn. Losing context is
annoying; losing the turn is the V1 outage mode.

Two quieter guards in the same spirit:

- **The UUID gate.** Transports mint non-UUID identities (`"cli-operator"`,
  `"discord:12345"`) until a resolver maps them. `_is_uuid` treats those as "no
  person" and skips the roundtrip entirely — no `::uuid` cast error every single turn.
- **No embedder = intentional degradation.** Without an embed seam there's no way to
  score memories against the query, so the memory half is skipped *on purpose* (not
  via the ValueError `retrieve_memories` would throw) and the profile half still
  stands.

## Config: two databases, two blast radii

`memories_database_url` is deliberately **separate** from `database_url` (the
checkpointer). "Durable threads" and "prod memories" are different blast radii — the
checkpointer may live in its own DB, and pointing the brain at the production
`aerys` database should be an explicit, single-purpose decision.

That connection is **READ-ONLY twice over**: the services are SELECT-only by contract
(doc 05), and `context_fn_for` sets `conn.read_only = True` at the session level — the
database itself will refuse a write even if a future bug tries one. While both brains
coexist, the n8n batch-extraction workflow (IfqY4BrhBGeQrcTC) stays the sole writer.

`embeddings_base_url` is a setting so a local embedding server is a `.env` change; the
model (`openai/text-embedding-3-small`, 1536-dim) MUST match what the write pipeline
stored in `memories.embedding`, or cosine distance compares apples to bananas.

## Assembly order and privacy default

Profile first (who they ARE — stable claims), memories second (what happened lately,
scored `similarity*0.7 + recency*0.3` against the current message) — the same order
the V1 prompt builder spliced them. Empty result = `''`, and the chat node injects
*nothing* rather than an empty `[What you know about this person]` header.

`privacy_context` defaults to `'private'` because today's only wired caller is the
owner's own channel (voice / HTTP behind the Bearer token). A future guild transport
passes `'public'` and the P-level/visibility gates in the services do the filtering —
the assembly logic doesn't change.

## The tests — 18 new, all offline

`FakeConn` replays canned rows (the pinned-Postgres-node trick from doc 05),
`ExplodingConn` proves the halves are fenced separately, and `RecordingModel` captures
the system prompt each turn to assert prompt *shape*: the block lands under the
header, `context_fn` receives `(person_id, latest user text)` — the latest human turn,
not history — and the capability claim appears exactly when retrieval is wired.
The HTTP tests prove owner mapping on both doors (`/ask` and `/v1/chat/completions`)
and that no `owner_person_id` keeps the anonymous `"http-caller"` behavior.

## Try it yourself

```bash
cd ~/projects/aerys-v2
uv run pytest -q                        # 267 green (18 new in test_context.py)
uv run pytest tests/test_context.py -v  # watch the graceful paths by name
```

Then live wiring is four `.env` lines and a restart of `--serve`:

```bash
MEMORIES_DATABASE_URL=postgresql://sira:***@192.168.1.231:5432/aerys
OWNER_PERSON_ID=6e6bcbed-03ef-4d17-95d2-89c467414335
EMBEDDINGS_API_KEY=sk-or-...            # OpenRouter; base_url defaults there
```

## What's deliberately NOT here yet

The **write path** — memory insertion, batch extraction, Guardian — stays in n8n; V2
reads what V1 writes. **Non-owner identity resolution** (Discord user → persons.id via
a real resolver) — the guild transport still mints `discord:<id>` strings that the
UUID gate correctly treats as "no person." **Connection pooling** — one connection per
turn is fine at personal-assistant volume (~1ms LAN roundtrip); a pool is a drop-in
swap behind the same seam. One seam at a time.
