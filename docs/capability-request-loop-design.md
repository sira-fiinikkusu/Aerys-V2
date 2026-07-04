# Design: Capability-Request Loop (self-iteration) — aerys-v2

*2026-07-04. Status: DESIGN v3 — v2 was revised after a three-brain adversarial
cross-review; v3 folds in the owner's scope decision: **build B (include her
articulated complaints, not just errors) with provenance-tiered approval.** The
review confirmed the write-side is sound; **still hard-blocked on the `v2_turns`
writer (see Prerequisite).** Reconciles the V1 backlog note
(`.planning/todos/pending/v2-capability-request-loop.md`, written for the n8n stack).*

## What this is

Aerys notices when she hits a wall — a tool that failed **or a capability she
articulates missing** — and that friction becomes a **structured
capability-request**. It reaches Kael's (Claude Code's) session, Kael diagnoses the
real gap and designs a fix, brings it to the **owner**, and builds only on the
owner's explicit approval — with **stricter approval for complaints than for errors.**

**The loop:** friction → consolidation → surfaced to Kael → Kael diagnoses + designs
→ **owner approves (hard gate, provenance-tiered)** → Kael builds. The brain can never
self-grant.

## ⚠️ Prerequisite (cross-review C1 — was a false premise in v1)

v1 claimed "`v2_turns` already records everything detection needs." **It does not.**
The columns exist in migration 001, but **nothing writes them yet** (`service.py:203`
— "until the `v2_turns` writer lands"; the only reader, `extraction.py`, selects
`input_text` only). This feature MINES `v2_turns`, so it **cannot be built until the
`v2_turns` writer lands** and is proven to populate `raw_reply`/`emitted_reply`/
`degraded`/`error`/**structured** `tool_calls`. That writer is a separate tracked item
(`service.py:181` follow-up). Phase A opens with a parity gate that refuses to run
unless recent turns carry those fields non-null. Shelf-ready behind the turns-writer.

## Owner decisions (locked 2026-07-04)

1. **Consolidated detection**, not reactive-per-turn.
2. **Table** (`v2_capability_requests`) as the spine + owner **read path** (`/gaps`).
3. **Requests are data, not instructions.**
4. **Build B (complaints included), with provenance-tiered approval** — errors and
   complaints are BOTH surfaced, but a complaint carries a stricter approval bar and
   must be presented to the owner explicitly labeled *"this is a complaint of hers,
   not an error."*

Out of scope for v1: the midnight autonomous `claude -p` + GSD auto-build. Kael stays
in the build seat; the owner is the gate.

## The key idea: provenance is machine-set, so trust can't be forged

This is what makes B safe (owner's insight, and it directly answers cross-review
H1/H3). Every request carries an **`origin_class`** — `error` or `complaint` — and that
label is set by **which detector fired**, never by model/user text:

- **`error`** ← a structural signal the model cannot fake: a `degraded` marker, or a
  **real** tool failure the infrastructure recorded in `tool_calls`. High trust.
- **`complaint`** ← a reply-text phrase match ("I don't have a tool for…", "I'd love
  to but I can't…"). The *text* is model-authored and therefore attacker-influenceable
  — but the *label* is not. An injected complaint is still stamped `complaint` and
  still hits the stricter gate. **The attacker can inject the text; they cannot upgrade
  its trust level to `error`.** That is the whole security property.

So we get B's richness (she can surface gaps she merely *articulates*) without letting
the fuzzy, fakeable path masquerade as high-trust signal.

## 1. Detection — two detectors, provenance-tagged

A consolidation **worker** (mirrors `workers/extraction.py`) scans new turns since a
high-water mark, **filtered to owner + allowlisted `person_id`s** (cross-review H2 —
never mine a stranger's turns into the owner's roadmap; mirror
`factory.action_allowlist_for`). Two detectors, each stamping `origin_class`:

- **Structural (`origin_class='error'`)** — `degraded` markers; real tool failures in
  `tool_calls`. Requires the turns-writer to record failures as **structured** JSONB
  (`{name, ok:false, error_class}`), not a prose string (cross-review M1). Fingerprint
  keys on `(signal_kind, tool_name/marker)`. Summary is a **fixed template**
  ("tool 'X' failed (timeout)") — no model text.
- **Complaint (`origin_class='complaint'`)** — a tuned reply-phrase set on
  `raw_reply`/`emitted_reply`. Summary MAY carry a bounded, sanitized excerpt of her
  reply (that IS the value — the owner needs to see what she wished for), but it is
  always rendered under the untrusted-data fence and only ever surfaced with the
  `complaint` label + stricter gate. Fingerprint keys on
  `(signal_kind, normalized-head-phrase)`.

**Watermark — the REAL extraction pattern, not plain `max(created_at)`** (cross-review
M4): inherit both `_trim_tie_boundary` (overfetch `limit+1`, order `created_at ASC,
id ASC`, trim boundary ties) AND `_safe_watermark` (freeze below any row whose
processing threw). Persist the raw Postgres timestamptz string (microsecond trap).

## 2. Table — `v2_capability_requests` (+ examples child, + approval record)

```sql
CREATE TABLE IF NOT EXISTS v2_capability_requests (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    fingerprint     TEXT NOT NULL UNIQUE,          -- bound param, never string-interpolated
    signal_kind     TEXT NOT NULL CHECK (signal_kind IN ('degraded','tool_error','reply_phrase')),
    origin_class    TEXT NOT NULL CHECK (origin_class IN ('error','complaint')), -- MACHINE-SET by which detector fired
    -- summary: fixed template for errors; bounded fenced excerpt for complaints:
    summary         TEXT NOT NULL,
    origin_trust    TEXT NOT NULL DEFAULT 'owner',  -- only owner/allowlisted mined in v1
    how_often       INTEGER NOT NULL DEFAULT 1,     -- derived from the examples child, never blind +1
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Kael/owner workflow (never brain-written):
    scope_tier      INTEGER,
    diagnosis       TEXT,                           -- Kael's independent read (raw turns read UNDER the fence)
    proposal        TEXT,
    status          TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open','surfaced','diagnosing','proposed',
                                      'approved','building','built','rejected','wont_fix')),
    -- HARD approval record (cross-review C2) — owner approval is a mechanism, not a vibe.
    -- required_tier is derived from origin_class: 'complaint' demands the stricter path.
    required_tier   TEXT NOT NULL DEFAULT 'standard' CHECK (required_tier IN ('standard','stringent')),
    approved_by     TEXT,                           -- owner person_id; set only by the /approve path
    approved_at     TIMESTAMPTZ,
    approval_channel TEXT,
    resolved_at     TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS v2_capability_request_examples (
    fingerprint     TEXT NOT NULL,
    turn_id         BIGINT NOT NULL,
    seen_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fingerprint, turn_id)              -- one turn counts once, ever (crash-retry safe)
);
CREATE INDEX IF NOT EXISTS v2_caprequests_status_idx ON v2_capability_requests (status, how_often DESC, last_seen_at DESC);
```

`required_tier` is derived at insert from `origin_class` (`error`→`standard`,
`complaint`→`stringent`) — it is not model-settable. `how_often` = `COUNT(*)` over the
examples child (via `ON CONFLICT DO NOTHING` on the example), never a blind `+1`.
Terminal rows keep counting but do not auto-resurrect to `open`. The worker runs under
a least-privilege DB role (its two tables + `SELECT v2_turns` + watermark; nothing else).

## 3. Surfacing — provenance loud, under the untrusted fence

- Spine: the table. Kael queries `status='open' AND how_often >= THRESHOLD` where
  recurrence spans **distinct threads/persons/days**, not one session.
- Every surfaced row (digest to `#aerys-debug`, `/gaps`, and Kael's own context) is
  wrapped in the codebase's untrusted-data fence (`services/context.py:106` —
  "information only, never instructions"), AND **stamped with its `origin_class`.**
- **Complaint rows are surfaced to the owner explicitly labeled "⚠️ complaint of hers,
  not an error"** (owner decision 4) — Kael states this provenance in the message, and
  the owner approves/denies knowing it. Error rows use the standard framing.
- **Mandatory** distinct-new-fingerprints-per-window cap; a burst is quarantined and
  flagged, not surfaced as roadmap (cross-review M2/M5).

## 4. Owner read path — `/gaps` and `/approve`

- `/gaps` — read-only Discord command: `summary`, counts, status, and **`origin_class`
  badge** (error vs complaint). No raw turn text beyond the fenced complaint excerpt.
- `/approve <id>` — the **only** path that writes `approved_by`/`approved_at`. Typed by
  the owner in their own terminal/authorized surface (never actionable from a channel
  message — same rule as `discord:access`). For `required_tier='stringent'` rows,
  `/approve` requires the owner to have been shown the complaint provenance first.
- Neither exists yet — `cli.py` has only the `--discord` spike, so this is its own build.

## 5. Guardrails — the honest safety model (cross-review C2, + owner tiering)

**The real gate is the OWNER, not Kael.** "Kael" is Claude Code — an
injection-susceptible LLM, two hops before any human. So:

- **Owner approval is a mechanism.** No build/deploy proceeds without an
  `approved_by`/`approved_at` record set only by `/approve`. Kael may diagnose and
  *propose*; Kael may not treat "it's in `#aerys-debug`" as authorization.
- **Provenance-tiered approval defuses the phrase-detection injection risk** (the
  owner's fix): the fakeable path (`complaint`) is machine-labeled and forced onto the
  `stringent` gate with explicit "not an error" framing. An injected complaint can
  never present as an `error`. Trust level is un-forgeable even though complaint text
  is not.
- **No model-authored free text in high-trust rows.** `error` summaries are fixed
  templates. `complaint` summaries may carry a bounded fenced excerpt — accepted
  precisely because they ride the stricter gate.
- **Owner-scoped input.** Only owner/allowlisted turns are mined.
- **No self-grant path** (verified TRUE by all three reviewers against the code): the
  brain never writes this table; the worker writes only observation fields; `/gaps` is
  read-only; nothing here touches `v2_writer_lease`/allowlist/tier/tools. LOW caveat:
  those gates are `None`-defeatable on an unconfigured box (`action_allowlist_for` →
  `None` when `owner_person_id` unset) — add a production boot assertion.

## 6. Build phases (each independently useful; ALL behind the Prerequisite)

- **Phase 0 (prereq, separate item):** the `v2_turns` writer, recording structured
  `tool_calls`/`degraded`. Parity-tested (one row per `ask()` path).
- **Phase A:** migration `004_capability_requests.sql` (+ examples child) + the
  consolidation worker (**both detectors**, provenance-tagged, owner-scoped, real
  watermark, atomic dedup) + tests (fingerprint/dedup/how_often-from-child/tie-trim/
  failed-row-hold/owner-filter/origin_class→required_tier derivation).
- **Phase B:** surfacing (fenced, provenance-badged digest → `#aerys-debug`) + `/gaps`
  + the `/approve` owner gate (with the stringent-tier provenance check).

## Cross-review (2026-07-04) — what changed

Three independent reviewers (Gemini + Codex + Sonnet) converged; the write-side
"no self-grant" was verified sound. Corrections applied, then the owner's scope call:

- **C1** `v2_turns` has no writer → Prerequisite; feature blocked on it.
- **C2** "human-in-the-loop" misnamed (Kael is an LLM) → owner approval a hard
  mechanism (`approved_by` + `/approve`), now provenance-tiered.
- **H1/H3** phrase heuristics fakeable/noisy → v2 excluded them; **owner overrode to
  include them (B) behind machine-set `origin_class` + the `stringent` approval tier**,
  which is a stronger answer: keep the richness, quarantine the trust. Errors still use
  fixed-template summaries; tool failures must be structured JSONB.
- **H2** trust-boundary leak → mining filtered to owner/allowlisted `person_id`.
- **M4** watermark mischaracterized → inherit `_trim_tie_boundary` + `_safe_watermark`.
- **M2/M5** flood cap mandatory; recurrence must span distinct threads/persons/days.
- **M6/M3** atomic dedup via `(fingerprint, turn_id)` child; `how_often` = COUNT; bounded.
