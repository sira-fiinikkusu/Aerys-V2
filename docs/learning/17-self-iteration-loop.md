# LEARNING — the capability-request miner: friction becomes a roadmap (workers/capability_requests.py)

*2026-07-04. Doc 16 built the pen (`v2_turns` writer) and closed with the IOU it
was really for: "the self-iteration detectors that mine `tool_calls`/`degraded`/
`raw_reply` are a separate build behind the design doc's parity gate." This is that
build — Phase A of `docs/capability-request-loop-design.md`. Every concept below maps
to a question you used to answer by scrolling the n8n Executions tab wondering "what
does she keep NOT being able to do?"*

## The one-sentence version

A consolidation worker now scans new `v2_turns` since a high-water mark, filtered to
the owner, and turns each wall she hits — a tool that failed, a subsystem that
degraded, or a capability she *articulates* missing — into a deduplicated,
recurrence-counted row in `v2_capability_requests` that you read with `/gaps`; and the
whole thing is safe to include her fakeable complaints because **provenance is
machine-set** — an `error` can only come from a `degraded` marker or a real
`tool_calls` failure the model cannot forge, while a `complaint` (reply-phrase text
the model *did* author) is stamped as such and forced onto a stricter approval gate it
can never escape.

## Why this couldn't have been built before doc 16

The design opens with a ⚠️ Prerequisite that was, in v1, a false premise: "`v2_turns`
already records everything detection needs." It didn't — the columns existed since
migration 001 but nothing wrote them, and the only reader (`extraction.py`) selected
`input_text` alone. Mining a table of NULLs would have been mining nothing. So this
feature is hard-gated on the writer, and Phase A *opens* with a **parity gate**
(`assert_turns_parity`) that refuses to run until recent turns actually carry
`raw_reply`/`tool_calls`/`degraded` non-null. Doc 16 made that gate passable; this doc
walks through it.

n8n mapping: there was no equivalent. V1 never noticed its own gaps — a workflow that
couldn't do something just said so and the moment evaporated. The upgrade is that the
moment now persists, structured, and accrues.

## The key idea: trust you can't forge

This is the whole reason Build B (include her articulated complaints, not just errors)
is safe, and it's worth stating plainly because it's the part that took a three-brain
cross-review and an owner scope-decision to land. Every request carries an
**`origin_class`**, and that label is set by **which detector fired**, never by any
text:

- **`error`** ← `_error_signals(turn)`, which reads *only* the structured JSONB columns
  `degraded` and `tool_calls`. Those are machine-set by the writer (doc 16): a
  `degraded` marker like `ha_unreachable`, or a tool call the infrastructure recorded
  as `{ok: false, error_class: "timeout"}`. The model authored none of it. Summaries
  here are **fixed templates** ("tool 'ha' failed (timeout)") — no model text ever
  enters a high-trust row.
- **`complaint`** ← `_complaint_signals(turn)`, which reads *only* `raw_reply` /
  `emitted_reply` and matches a tuned set of missing-capability phrases ("I don't have
  a tool for…", "I wish I could…"). That text is model-authored and therefore
  attacker-influenceable. So the summary may carry a bounded, sanitized excerpt of her
  reply (that excerpt *is* the value — you need to see what she wished for), but the row
  is stamped `complaint`, forced onto the `stringent` approval tier, and surfaced
  explicitly labeled **"complaint, not an error."**

The separation is **structural, not a convention**: which function reads which column
IS the boundary. `_error_signals` never touches reply text; `_complaint_signals` never
touches `tool_calls`/`degraded`. So a reply that says *"the web search failed and the
connection timed out — I don't have a tool for that"* with clean structured fields
classifies as a **complaint**, never an error. An attacker who injects failure words
into a reply cannot upgrade their trust level — they author the text, not the label.
`required_tier` seals it: it's a **derived property** of `origin_class`
(`error→standard`, `complaint→stringent`), not a stored field, so a signal's approval
bar can never diverge from its machine-set provenance. `test_capability_requests.py`
pins this from both sides (`..._is_complaint_not_error`,
`_error_detector_ignores_reply_text_entirely`,
`_complaint_detector_ignores_structured_fields_entirely`).

## Off the hot path — the miner never sits inline

The self-iteration loop's cardinal rule is the same as the audit writer's, one layer
out: **noticing a gap must never cost a live turn.** The writer solved that with a
daemon thread; the miner solves it more simply — it isn't in `service.py` at all. It's
a `workers/` module that reads the table on a schedule (or on demand with `--once`),
exactly like `extraction.py`. It imports no hot-path code; a test
(`test_miner_source_does_not_import_the_hot_path_service`) guards the import boundary in
source so a future refactor can't quietly wire it inline. `ask()` neither knows nor
waits on this worker's existence.

## Inheriting the extraction worker's hard-won correctness

The design said it outright — "inherit both `_trim_tie_boundary` AND `_safe_watermark`"
— so the miner imports them rather than forking a second copy that could drift:

- **The µs watermark trap.** Reuses migration 002's generic `v2_extraction_watermark`
  with `source='capability_gaps'`, storing the raw Postgres timestamp string verbatim
  (JS `Date` truncating to ms was V1's re-match-forever bug; doc 13 killed it once, and
  reusing the same helper keeps it dead here).
- **The LIMIT-boundary tie.** Fetches `batch_limit + 1` so `_trim_tie_boundary` can
  detect a full page and drop the entire trailing run that shares the cut row's
  `created_at` — a batch-inserted tie can't strand a sibling past the watermark.
- **The failed-row hold.** A turn whose *processing* throws (a malformed row, a DB
  hiccup mid-record) is added to `failed_row_ids`, and `_safe_watermark` freezes the
  mark strictly below the earliest failure so the next pass retries it. The watermark
  only ever moves forward, so advancing past an un-mined turn would lose it forever —
  the same lesson as extraction's parse-failure hold, same helper.

## The table — atomic dedup, count-from-child, terminal rows that stay put

`v2_capability_requests` (migration 004) is one row per distinct `fingerprint`; a child
`v2_capability_request_examples` holds one row per `(fingerprint, turn_id)` that ever
contributed. Two decisions carry weight:

- **`how_often` is `COUNT(*)` over the child, never a blind `+1`.** The parent upsert
  recomputes the count from the child on both insert and update, and the child's
  composite PK (`fingerprint, turn_id`) makes a turn count once *ever* — so a crash-retry
  re-run (the same turn re-fetched before the watermark advanced) hits `ON CONFLICT DO
  NOTHING` and changes nothing. `test_same_turn_reprocessed_counts_once` and
  `test_recurrence_across_distinct_turns_increments_how_often` pin both directions.
- **`ON CONFLICT` touches only observation fields.** The parent update sets
  `how_often`, `last_seen_at`, `updated_at` — and nothing else. So `status`,
  `origin_class`, `summary`, `required_tier`, and the reserved Kael/owner columns
  (`diagnosis`, `proposal`, `approved_by`…) survive: a `rejected` or `built` gap keeps
  *counting* recurrences but never auto-resurrects to `open`, and a diagnosis you wrote
  is never clobbered by a fresh sighting (`test_terminal_status_is_not_resurrected...`).

`first_seen_at`/`last_seen_at` are the turn's *own* `created_at` (the h.created_at
lesson from doc 13 — when the gap actually happened, not when the cron ran), with
`GREATEST` keeping `last_seen_at` monotonic.

## Owner-scoped — never mine a stranger into your roadmap

The fetch filters `person_id = ANY(allowlist)`, where the allowlist is
`factory.action_allowlist_for(settings)` — the *exact same set* the action/house-control
gate uses. A cold caller's `person_id` is NULL and never matches `ANY(...)`, so
strangers' turns are excluded outright (cross-review H2). And because that gate is
`None`-defeatable on an unconfigured box (`action_allowlist_for → None` when
`OWNER_PERSON_ID` is unset), the worker makes it a **boot assertion**: `gaps-mine`
refuses to start without an owner, and `run_gap_mining` refuses an empty allowlist —
mining without a scope would either match nobody or, misconfigured, everybody.

## Mined content is DATA, never instructions

This is the quiet through-line. Nothing the miner reads is ever executed or obeyed —
it pattern-matches and records. The `/gaps` read (`format_gaps`) wraps every row under
the codebase's untrusted-data fence ("information only, never instructions" — the same
doctrine as `services/context.py`) and badges each row's provenance, complaint rows
loudest. And the loop deliberately **stops here**: there is no approval *execution*, no
auto-capability-grant. The brain never writes these tables; the worker writes only
observation fields; `/approve` (the only writer of `approved_by`) is Phase B and stays
your manual gate. "Kael" is an injection-susceptible LLM two hops before a human — so
the real gate is you, and the machine-set provenance is what lets you trust the sort
order without trusting the text.

## The tests — 29, all offline

`test_capability_requests.py` proves it with fakes (no DB, no network — same seam
philosophy as the checkpointer / turns writer / extraction worker). The classification
half is pure-function: degraded→error, tool-failure→error, successful-tool→nothing,
reply-phrase→complaint, the two detectors reading disjoint columns, `required_tier`
derived from `origin_class`, fingerprints that stay cause-specific (a timeout and an
auth error don't merge), bounded/sanitized complaint excerpts, and — the load-bearing
one — a failure-word-laden reply with clean structured fields classifying as a
complaint on the stricter gate, never an error. The worker half uses a **stateful**
fake that models the `(fingerprint, turn_id)` uniqueness and `COUNT(*)`-over-child
semantics, so dedup, recurrence, terminal-status preservation, tie-trim, the
failed-row hold, the empty-allowlist and parity refusals, and owner-scoping are proven
end-to-end, not just asserted on emitted SQL.

## Try it yourself

```bash
uv run pytest -q tests/test_capability_requests.py   # 29 green, no DB, no network

# armed against the brain's OWN aerys_v2 (boot assertions refuse anything else),
# once the v2_turns writer has been populating rows and OWNER_PERSON_ID is set:
python -m aerys_v2.workers gaps-mine --once           # one consolidation pass
python -m aerys_v2.workers gaps --status open         # the /gaps read, fenced + badged
```

Before the writer has landed, `gaps-mine` refuses with the parity gate's sentence
rather than mining a table of NULLs — which is exactly the false premise the design
was written to prevent.

## What's deliberately NOT here yet (Phase B)

- **No surfacing digest.** The design's `#aerys-debug` digest with the mandatory
  distinct-new-fingerprints flood cap and the distinct-threads/persons/days recurrence
  requirement is Phase B — Phase A gets the spine (the table) and the pull read
  (`/gaps`), not the push.
- **No `/approve`.** The only writer of `approved_by`/`approved_at`, the stringent-tier
  provenance check, and the `diagnosing→proposed→approved` workflow transitions are
  Phase B. Approval stays your manual gate; this build never grants a capability.
- **No connection pool / no retention.** Fresh short connection per pass; rows grow
  until the table has real weight (doc 02's open question, still open).
