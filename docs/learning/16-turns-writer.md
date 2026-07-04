# LEARNING — the v2_turns writer: the audit spine gets its pen (turns.py + service.py)

*2026-07-04. Doc 02 shipped the table empty ("No writers. Nothing inserts into
v2_turns/v2_outbox yet"). Docs 13 and 15 then kept writing the same footnote — "the
router log IS the persistence for the tier decision until the v2_turns writer lands."
This is that writer. Every concept below maps to a question you used to answer by
clicking through the n8n Executions tab.*

## The one-sentence version

Every completed `ask()` turn now writes exactly one row to `v2_turns` — on EVERY exit
path (chat, action, voice-chat, voice-action, timeout, and the raises), strictly AFTER
the reply is in hand, on a daemon thread the transport never waits on, and fail-open so
a down NAS can never crash a live turn — which finally makes doc 02's forensic table
real and unblocks the one feature that's hard-gated on it (`docs/capability-request-loop-design.md`).

## The IOU comes due

Three docs deep, every "until then" was the same promise:

- Doc 02 designed the columns and shipped the migration, deliberately empty, so no
  earlier canary would invent an incompatible local shape.
- Doc 13's extraction worker already READS `v2_turns` (her own conversations feed
  memory) — but only `input_text`. The audit table earned its keep once; the rest of
  the row was still `NULL`.
- Docs 13 and 15 both ended with the tier decision living only in a log line, "same
  fields the turns row will carry" — the fields existed, nothing filled them.

`turns.py` + the `service.py` recording seam pay all three. The router log stops being
the persistence; the Executions tab doc 02 called "the execution list you actually own"
becomes a table something actually writes.

## What each column captures — and who fills it

The row is assembled in one place (`build_turn_row`), the INSERT lives beside it
(`INSERT_TURN_SQL`) so columns and dict keys can't drift, and `service.py` supplies the
per-path values. The interesting ones:

| Column | What the writer puts there | Path-specific truth |
|---|---|---|
| `channel` | `derive_channel(thread_id)` — the checkpointer prefix every transport already sets | `voice`/`discord_dm`/`guild`/`telegram_*`/`cli`/`http`; NOT NULL, so an unknown shape degrades to the head token, never raises |
| `person_id` vs `platform_identity` | a resolved `persons.id` (a real UUID) lands in `person_id`; a cold `"platform:id"` handle lands in `platform_identity`, `person_id` NULL | the same `_is_uuid` split as `services.context` — the UUID column stays clean, the cold caller is still on the record |
| `classifier_intent` / `tier` | `"chat"`/`"action"` and the tier that ACTUALLY served | chat-only path (no router) records both NULL — the row states what happened, not a decision never made; voice records `tier="standard"` (the ChannelPolicy pin), never the router's ignored hint |
| `tier_override_source` | why the served tier differs from the classifier's pick | `"deep_cap"` when the daily cap downgraded deep→standard (doc 15) — the column comment gained that value in this change |
| `raw_reply` / `emitted_reply` | the model's output vs what the channel actually got | equal on chat (no polish step — V1's Gemini polisher is now prompt-side emotion tags); the split earns its keep on voice-action (below) |
| `latency_ms` / `trace_id` | wall time and the active Phoenix span id | captured SYNCHRONOUSLY, in-context (see below) |
| `error` | the timeout message, or the exception text on a raised turn | a turn that ran past budget or blew up is exactly what forensics need — so those paths audit too |

`guard_verdict` and `resolver_version` are in the schema but the writer passes neither
yet — the guard and resolver-version plumbing land with their own phases, and the row
records an honest NULL rather than a fabricated value.

## Structured, not prose — the load-bearing choice

Two columns are JSONB and MUST stay structured, because a downstream feature mines them
by machine: `tool_calls` (`list[{name, ok: bool, error_class: str|null}]`) and
`degraded` (`list[str]` of markers like `["ha_unreachable"]`). A prose string in either
defeats the whole self-iteration loop (cross-review M1). So `turns.py` does real work to
build them, not `str()`:

- **`extract_tool_calls`** walks the action subgraph's own message list — one entry per
  `ToolMessage`, in execution order, name resolved from `.name` with a fallback map
  through the requesting `AIMessage.tool_calls` (belt-and-braces: `.name` has been None
  across langchain-core versions; an unresolvable name collapses to `"unknown"` and logs
  a warning so the blind merge is visible, not silent).
- **`classify_tool_result`** turns a call into `ok`/`error_class`. Two signal sources,
  ranked by trust: `status=="error"` is LangChain's authoritative "the tool raised" —
  its machine-set message is refined into a cause (`timeout` vs `auth_error` vs
  `unreachable` don't merge). Otherwise the action-stack tools NEVER raise by design (a
  raise inside a ToolNode kills the whole turn — the V1 failed-webhook outage), so a real
  failure comes back as an honest STRING, matched against a curated sentinel table.
- **The anti-forgery boundary** (`_first_line`, cross-review sharp-3 H): only the FIRST
  line of a tool's content is scanned. A tool's honest failure sentinel always leads its
  content; a success payload's body (a web snippet, a document that happens to contain
  the words "web search failed") rides on line 2+. So attacker-echoed text can never be
  substring-matched into a forged failure, and a real failure is never missed. Refusals
  ("not on the allowlist") and empty results ("returned no results") stay `ok=True` —
  the tool did its job.

This is the difference between an audit log you can read and one a program can act on.
The self-iteration loop keys its high-trust `origin_class='error'` on exactly these two
machine-set fields — the model authors the reply text, but it cannot author a `degraded`
marker or a `tool_calls` failure. That un-forgeable-ness is the whole security property
downstream, and it only exists because the writer refuses to flatten them to prose.

## Two hard rules: after the response, and fail-open

An audit row must never cost the live turn anything — not a byte of latency, not a
single failure mode. Both rules are structural.

**Off the hot path, but built in-context.** The row is BUILT synchronously inside the
turn (`_fire_turn_record`), then WRITTEN on a daemon thread. The split is deliberate and
load-bearing: `trace_id`, `tool_calls`, and `latency_ms` can only be captured *here*,
with the data in hand and inside doc 11's `_turn_span` — a background thread has no
current OTel span and no reply object. So we snapshot synchronously (microseconds of
pure-Python work) and hand the finished dict to `threading.Thread(daemon=True).start()`.
The reply returns to the transport without ever waiting on the NAS insert; a slow NAS
costs a lingering background thread, never the user's latency. `test_record_off_hot_path_does_not_block_return`
pins it: a recorder that `sleep(5)`s still returns the reply in under 2s.

**Fail-open, at every layer.** This mirrors AlertSink's doctrine from doc 02 —
*alerting about an error must not become the error* — applied to auditing: recording a
turn must not become the turn's failure. So:

- `record_turn=None` (no `DATABASE_URL`, dev/CI) short-circuits the whole thing —
  byte-for-byte the old behavior.
- The recorder factory (`turn_recorder_for`) opens a fresh short connection per turn with
  `connect_timeout=5` and `statement_timeout=5000` — a DOWN NAS can't hang on TCP-SYN for
  ~127s, a SLOW one can't hold an `aerys_v2` slot open against `max_connections` and
  starve the hot path's own DB access. Any DB trouble is logged and swallowed.
- Row building is wrapped; a serialization bug drops the row, never the turn.
- A `BoundedSemaphore(32)` fuse caps concurrent audit threads. At personal-assistant
  volume it's never neared — it exists so that under a NAS outage, threads-plus-inserts
  pile toward `RLIMIT_NPROC` and Postgres slots; over the cap we DROP the write
  (fail-open) rather than march the box into the ground.
- Even `Thread.start()` is guarded — it was the ONE audit-path line outside a
  try/except (cross-review hotpath H), and under thread exhaustion it raises
  `RuntimeError`; unguarded, that unwinds into the live reply. Now a failed spawn
  releases the semaphore and logs. The audit may lose a row; the turn may not.

**Every exit, including the raises.** The docstring promise — "a row on EVERY completion
path" — is only true if the error exits audit too, and they do: a chat/action/voice
`invoke()` that raises fires a failure row (`error` = the exception text, `degraded` +=
`turn_failed` or `recursion_limit` for a rail trip) BEFORE re-raising unchanged
(cross-review correctness H). A turn that blew up is the highest-value row the
self-iteration loop will ever read; losing it would be the exact wrong place to be quiet.

## raw_reply vs emitted_reply — the split that pays off in voice

On chat the two are equal, so the column pair looks redundant. Voice-action is where it
was designed for: the caller HEARS the ack ("[warmly] On it") long before the tool loop
finishes, so `emitted_reply` is the ack and `raw_reply` is the action's real outcome
("Light's off"). The audit records both — what she said out loud AND what actually
happened — and the write fires from INSIDE the already-background completion thread, off
the hot path by construction (the ack shipped seconds ago). `test_voice_action_path_records_ack_as_emitted_and_final_as_raw`
pins exactly this.

## The payoff — the self-iteration loop's hard prerequisite

This is why the writer got built now. `docs/capability-request-loop-design.md` — Aerys
noticing when she hits a wall and turning that friction into a structured
capability-request for Kael — opens with a ⚠️ Prerequisite that names this exact gap:

> v1 claimed "`v2_turns` already records everything detection needs." **It does not.**
> The columns exist in migration 001, but nothing writes them yet… This feature MINES
> `v2_turns`, so it cannot be built until the `v2_turns` writer lands and is proven to
> populate `raw_reply`/`emitted_reply`/`degraded`/`error`/**structured** `tool_calls`.

Its whole safety model rests on provenance that can't be forged: `origin_class='error'`
comes from a `degraded` marker or a real `tool_calls` failure — machine-set signals the
model cannot fake — while `origin_class='complaint'` comes from fakeable reply-phrase
text and is forced onto a stricter approval gate. That distinction is only coherent
because this writer records the structural signals structurally. The writer is Phase 0
of that design; its Phase A opens with a parity gate that refuses to run until recent
turns carry those fields non-null. We just made that gate passable.

## The tests — 40, all offline

`test_turns.py` proves the row shape with fakes (no DB, no network — same seam
philosophy as the checkpointer / speak_fn / router): one row per completion path (chat,
action, voice-chat, voice-action, deep-cap downgrade, timeout, and each raise);
`tool_calls`/`degraded` structured not prose; `person_id` vs `platform_identity`;
channel derivation for every transport prefix; the anti-forgery boundary (a success
payload echoing "web search failed" stays `ok=True`); exception-cause refinement;
`ha_write_failed` distinct from `ha_unreachable`; a drift-guard that breaks CI if a tool
reword falls out of the sentinel table; and the fail-open trio — a raising recorder, a
dead-NAS factory, a spawn-failure, and the in-flight-cap drop — none of which touch the
turn.

## Try it yourself

```bash
uv run pytest -q tests/test_turns.py          # 40 green, no DB, no network
# unarmed (no DATABASE_URL) — logs "v2_turns audit UNRECORDED", turns run unchanged:
uv run aerys-v2 --serve
# armed against the brain's OWN aerys_v2 (boot assertions from doc 15 refuse anything else),
# then read a turn back as SQL — the question that used to mean clicking an execution:
psql "$DATABASE_URL" -c \
  "SELECT thread_id, tier, tier_override_source, tool_calls, degraded, latency_ms
     FROM v2_turns ORDER BY id DESC LIMIT 5;"
```

## What's deliberately NOT here yet

- **No outbox linkage.** The turn records what WAS emitted; the outbox (doc 02) records
  the intent-to-emit with idempotency. `v2_outbox.turn_id` stays NULL until the emit path
  writes through the outbox — the INSERT doesn't `RETURNING id` yet because nothing needs
  the id back.
- **No new reader.** Extraction still selects only `input_text`; the self-iteration
  detectors that mine `tool_calls`/`degraded`/`raw_reply` are a separate build behind the
  design doc's parity gate.
- **No connection pool.** Fresh short connection per turn — a personal-assistant volume
  never strains it; the pool is a drop-in later (same note as `context_fn_for`).
- **No retention policy.** Turns grow forever until the table has real weight (doc 02's
  open question — decided then, not now).
- **`guard_verdict` / `resolver_version` unfilled.** The columns wait for the guard and
  resolver-version phases; the writer records NULL rather than guess.
