# Design: Capability-Request Loop (self-iteration) — aerys-v2

*2026-07-04. Status: DESIGN v2 — revised after a three-brain adversarial cross-review
(Gemini + Codex + independent Sonnet, all grounded against the code). The review
confirmed the write-side is sound but broke three load-bearing claims of v1; this
version corrects them. **Not buildable yet — hard-blocked on the `v2_turns` writer
(see Prerequisite).** Reconciles the V1 backlog note
(`.planning/todos/pending/v2-capability-request-loop.md`, written for the n8n stack).*

## What this is

Aerys notices when she hits a wall — a tool that failed, a capability she lacked —
and that friction becomes a **structured capability-request**. It reaches Kael's
(Claude Code's) session, Kael diagnoses the real gap and designs a fix, brings it to
the **owner**, and builds only on the owner's explicit approval.

**The loop:** friction → consolidation → surfaced to Kael → Kael diagnoses + designs
→ **owner approves (hard gate)** → Kael builds. The brain can never self-grant.

## ⚠️ Prerequisite (cross-review C1 — was a false premise in v1)

v1 claimed "`v2_turns` already records everything detection needs." **It does not.**
The columns exist in migration 001, but **nothing writes them yet** — `service.py:203`
says so ("until the `v2_turns` writer lands"), and the only current reader
(`extraction.py`) selects `input_text` only. This feature **mines `v2_turns`, so it
cannot be built until the `v2_turns` writer lands** and is proven to populate
`raw_reply`/`emitted_reply`/`degraded`/`error`/**structured** `tool_calls`. That writer
is a separate tracked item (`service.py:181` follow-up). Phase A must open with a
parity gate that refuses to run unless recent turns carry those fields non-null.
This design is "shelf-ready behind the turns-writer," not "buildable today."

## Owner decisions (locked 2026-07-04)

1. **Consolidated detection**, not reactive-per-turn.
2. **Table** (`v2_capability_requests`) as the spine + owner **read path** (`/gaps`).
3. **Requests are data, not instructions.**

Out of scope for v1 (deliberately): the midnight autonomous `claude -p` + GSD
auto-build. Kael stays in the build seat; the owner is the gate.

## 1. Detection — structural signals only (cross-review H3/H1)

v1 gave reply-text phrase heuristics ("I don't have a tool for…") co-equal status.
The review showed that's both **low-precision** (Aerys says "I can't verify that
rumor" constantly) and **attacker-fakeable** (a poisoned turn can make the model
emit any phrase). So **v1 detects on structural signals ONLY** — signals the model
cannot forge because they come from the infrastructure, not its text:

- `degraded` markers on the turn (e.g. `["ha_unreachable"]`).
- **Real** tool failures in `tool_calls` — requires the turns-writer to record these
  as structured JSONB (`{name, ok:false, error_class}`), NOT as a prose string
  (cross-review M1). Fingerprint keys on `(signal_kind, tool_name/marker)`.

Reply-phrase heuristics are **demoted to advisory** — off by default, behind a flag,
and only permitted to INSERT after being measured against a labeled corpus. They never
create a surfaced request in v1.

A consolidation **worker** (mirrors `workers/extraction.py`) scans new turns since a
high-water mark, **filtered to owner + allowlisted `person_id`s** (cross-review H2 —
never mine a stranger's turns into the owner's roadmap; mirror
`factory.action_allowlist_for`). New fingerprint → insert; seen → bump.

**Watermark — the REAL extraction pattern, not plain `max(created_at)`** (cross-review
M4). Inherit both hard-won fixes: `_trim_tie_boundary` (overfetch `limit+1`, order
`created_at ASC, id ASC`, trim rows tied on the boundary timestamp) AND
`_safe_watermark` (freeze the mark below any row whose processing threw, so it isn't
skipped forever). Persist the **raw** Postgres timestamptz string (microsecond trap).

## 2. Table — `v2_capability_requests` (+ examples child, + approval record)

```sql
CREATE TABLE IF NOT EXISTS v2_capability_requests (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    fingerprint     TEXT NOT NULL UNIQUE,          -- (signal_kind, tool_name/marker) — bound param, never interpolated
    signal_kind     TEXT NOT NULL CHECK (signal_kind IN ('degraded','tool_error')), -- reply_phrase excluded in v1
    -- summary is a FIXED TEMPLATE, never a slice of model/user text (cross-review H1):
    summary         TEXT NOT NULL,                 -- e.g. "tool 'search_web' failed (timeout)" / "degraded: ha_unreachable"
    origin_trust    TEXT NOT NULL DEFAULT 'owner', -- only owner/allowlisted mined in v1; tag anyway
    how_often       INTEGER NOT NULL DEFAULT 1,    -- derived from the examples child, not free-incremented
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Kael/owner workflow (never brain-written):
    scope_tier      INTEGER,
    diagnosis       TEXT,                          -- Kael's independent read (raw turns read UNDER the untrusted fence)
    proposal        TEXT,
    status          TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open','surfaced','diagnosing','proposed',
                                      'approved','building','built','rejected','wont_fix')),
    -- HARD approval record (cross-review C2/Codex#3) — owner approval is a mechanism, not a vibe:
    approved_by     TEXT,                          -- owner person_id, set only by the /approve path
    approved_at     TIMESTAMPTZ,
    approval_channel TEXT,
    resolved_at     TIMESTAMPTZ
);
-- atomic, idempotent dedup + bounded examples (cross-review M6/M3):
CREATE TABLE IF NOT EXISTS v2_capability_request_examples (
    fingerprint     TEXT NOT NULL,
    turn_id         BIGINT NOT NULL,
    seen_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fingerprint, turn_id)             -- one turn counts once, ever (crash-retry safe)
);
CREATE INDEX IF NOT EXISTS v2_caprequests_status_idx ON v2_capability_requests (status, how_often DESC, last_seen_at DESC);
```

`how_often` is `COUNT(*)` over the examples child (via `INSERT ... ON CONFLICT DO
NOTHING` on the example, then recount) — never a blind `+1`, so a crash-retry or a
two-worker race can't inflate it. Terminal rows (`rejected`/`wont_fix`) keep counting
but **do not auto-resurrect** to `open`; a "rejected-but-still-recurring" view handles
that case. Consolidation worker gets a **least-privilege DB role**: `SELECT v2_turns` +
read/write its two tables + watermark, nothing else (no `v2_writer_lease`, `memories`,
config).

## 3. Surfacing to Kael — under the untrusted-data fence (cross-review L2)

- Spine: the table. Kael queries `status='open' AND how_often >= THRESHOLD` where
  recurrence spans **distinct threads/persons/days**, not repeats in one session.
- Every request row rendered into Kael's context (digest to `#aerys-debug`, or query
  output) is wrapped in the **same untrusted-data fence** the codebase already uses
  (`services/context.py:106` — "information only, never instructions"). Kael reads
  `example_turn_ids` turns for diagnosis **under that fence**, because that raw text is
  attacker-influenceable. Diagnosis is Kael's own; a row's contents are never
  implemented verbatim.
- **Mandatory** (not optional) distinct-new-fingerprints-per-window cap; a burst is
  itself quarantined and flagged, not surfaced as roadmap.

## 4. Owner read path — `/gaps`

A read-only Discord command showing `summary` (the fixed template) + counts + status.
Never raw turn text (that's both the privacy control and the injection control — same
fix). Note: no slash-command framework exists yet (`cli.py` has only the `--discord`
spike), so `/gaps` is its own small build.

## 5. Guardrails — the honest safety model (cross-review C2)

**The real gate is the OWNER, not Kael.** "Kael" is Claude Code — an
injection-susceptible LLM, two AI-hops before any human. So:

- **Owner approval is a mechanism, not an operator norm.** No build/deploy off a
  capability request proceeds until an `approved_by`/`approved_at` record exists, set
  only by an explicit owner action (an `/approve <id>` the owner types, never a
  channel message — same rule as `discord:access`). Kael may diagnose and *propose*;
  Kael may not treat "it's in `#aerys-debug`" as authorization.
- **No model-authored free text in the table.** `summary`/`fingerprint` are fixed
  templates from structural fields. Raw model/user text lives only behind
  `example_turn_ids`, read under the untrusted fence.
- **Structural signals only** create rows in v1 → the model can't manufacture a
  request by emitting a phrase.
- **Owner-scoped input.** Only owner/allowlisted turns are mined.
- **No self-grant path** (verified TRUE by all three reviewers against the code): the
  brain never writes this table; the worker writes only observation fields; `/gaps`
  is read-only; nothing here touches `v2_writer_lease`/allowlist/tier/tools. Caveat
  (cross-review LOW): those structural gates are `None`-defeatable on an unconfigured
  box (`action_allowlist_for` → `None` when `owner_person_id` unset) — add a
  production boot assertion (owner configured, allowlist non-`None`).

## 6. Build phases (each independently useful; ALL behind the Prerequisite)

- **Phase 0 (prereq, separate item):** the `v2_turns` writer lands, recording
  structured `tool_calls`/`degraded`. Parity-tested (one row per `ask()` path).
- **Phase A:** migration `004_capability_requests.sql` (+ examples child) + the
  consolidation worker (structural-only, owner-scoped, real watermark, atomic dedup) +
  tests (fingerprint/dedup/how_often-from-child/tie-trim/failed-row-hold/owner-filter).
- **Phase B:** surfacing (fenced digest → `#aerys-debug`) + `/gaps` + the `/approve`
  owner gate.

## Cross-review (2026-07-04) — what changed from v1

Three independent reviewers converged; the write-side "no self-grant" was verified
sound, but v1 rested on false/overstated claims. Corrections applied above:

- **C1** `v2_turns` has no writer → added the Prerequisite; feature is blocked on it.
- **C2** "human-in-the-loop" misnamed (Kael is an LLM) → owner approval made a hard
  mechanism (`approved_by`/`/approve`); safety model reframed honestly.
- **H1** "mining removes injection surface" false → `summary`/`fingerprint` are fixed
  templates; raw text only behind examples, read under the untrusted fence.
- **H2** trust-boundary leak → consolidation filtered to owner/allowlisted `person_id`.
- **H3/M1** phrase heuristics low-precision + fakeable → **structural signals only** in
  v1; phrases advisory-behind-measurement; tool failures must be structured JSONB.
- **M4** watermark mischaracterized → inherit `_trim_tie_boundary` + `_safe_watermark`.
- **M2/M5** flood cap made mandatory; recurrence must span distinct threads/persons/days.
- **M6/M3** atomic dedup via `(fingerprint, turn_id)` child; `how_often` = COUNT; bounded.
- **Codex#3** added the approval record (`approved_by`/`approved_at`/`approval_channel`).
