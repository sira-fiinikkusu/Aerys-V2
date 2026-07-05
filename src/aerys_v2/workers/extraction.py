"""Memory extraction worker — n8n 04-02 Batch Extraction (IfqY4BrhBGeQrcTC) in SHADOW MODE.

The port, node by node:
  Read Last Processed        -> read_watermark()   (v2 table, not $getWorkflowStaticData)
  Fetch from n8n_chat_histories -> PROD_MESSAGES_SQL (query ported verbatim, + raw ts)
  Group Messages             -> group_by_person()  (person_id FIRST — the pre-05.1
                                misattribution lesson: never let one transcript's
                                facts land under every person_id in the room)
  Build Extraction Request   -> build_transcript() + EXTRACTION_SYSTEM_PROMPT (verbatim)
  Call LLM for Extraction    -> the injected `llm` seam (openrouter_chat() live)
  Parse Observations         -> parse_observations() + _compose_content()
  Embed Observation          -> the injected `embedder` seam (same model as retrieval:
                                services.memory.EMBED_MODEL, 1536-dim — MUST match or
                                cosine distance compares apples to bananas)
  Insert Memory              -> INSERT_STAGING_SQL — into v2_memories_staging, NEVER
                                prod memories. Shadow mode also drops the whole
                                Dedup Check / Soft Delete / Update branch: staging is
                                append-only, diffing raw extractions is the point.

New vs v1: a SECOND source. Her own V2 conversations are read from v2_turns (the
audit spine, migration 001) — NOT the LangGraph checkpoint blobs, because identity
never lands in graph state (test_identity_never_lands_in_state proves it), so
checkpoints are person-blind and grouping by person_id would be impossible there.
v2_turns carries person_id + input_text + created_at, exactly what extraction needs.

n8n bugs fixed in the port:
  - High-water mark stored `new Date().toISOString()` — JS Date truncates to
    milliseconds, timestamptz keeps microseconds, so `>` re-matched the same row
    forever; v1 papered over it with a +1ms bump. Here the RAW Postgres string
    (created_at::text) is stored verbatim and round-trips exactly — no bump needed.
  - `ORDER BY h.id DESC LIMIT 200` fetched the NEWEST 200: with a >200-row backlog
    the watermark jumped past the overflow and those rows were never extracted.
    Both queries here are ORDER BY created_at ASC — the watermark only ever
    advances through rows actually processed.
  - Memory created_at: original message time (the batch's latest created_at), not
    NOW() — recency scoring needs "when did this happen", not "when did the cron run".

Live mode (run_live_extraction / --live): same read -> group -> LLM -> embed
pipeline, but observations land in PROD `memories` via triage_memory() instead
of v2_memories_staging — the "Insert Memory" node's real dedup branch, not the
shadow's append-only stand-in:
  Dedup Check   -> SELECT the live (person_id, key_label) row, if any
  (branch)      -> value unchanged: UPDATE content/context/event_date/embedding
                -> value changed:   soft-delete old row + INSERT new, atomically
                   (the CTE pattern from CLAUDE.md's "atomic row replacement")
                -> no existing row: plain INSERT
Three hard gates run BEFORE any read or LLM spend, mirroring run_boot_assertions'
refuse-loudly stance applied to a single pass: the n8n batch-extraction
workflow (IfqY4BrhBGeQrcTC) must be INACTIVE, v2_writer_lease
(kind='memory_extraction') must be held by 'brain', AND this pass must
actually acquire the write mutex (acquire_write_mutex, a real
pg_try_advisory_xact_lock — the lease ROW only says who's authorized, it does
nothing to stop two 'brain' processes racing each other) — two armed writers
on the same (person_id, key_label) unique index is a race, not a feature.

Adversarial-review fixes (2026-07-04), all in this file unless noted:
  - parse_observations() now distinguishes a genuine empty extraction ([])
    from a parse failure (None) — max_tokens=1200 makes truncation routine on
    a busy transcript, and a truncated reply used to look identical to "no
    memories here", silently advancing the watermark past lost messages.
    Both run_extraction and run_live_extraction now hold the watermark below
    any parse-failed group's messages (_safe_watermark) instead.
  - Both source queries are fetched as `batch_limit + 1` and passed through
    _trim_tie_boundary — a >batch_limit backlog whose cut line landed mid-tie
    (two rows sharing one microsecond-identical created_at, real since
    now() is transaction-time) used to strand the tied sibling(s) forever.
  - triage_memory() now refuses (returns 'skipped') any observation with a
    NULL/blank key_label (unusable for the (person_id, key_label) dedup index
    — was landing as unbounded duplicate INSERTs) or a blank value_text (was
    scoring 0.0 similarity against anything real, forcing a REPLACE that
    soft-deleted a good memory for a blank one).
  - TRIAGE_UPDATE_SQL now writes privacy_level too, but only to TIGHTEN
    (public -> private) — a same-value restatement re-tagged private no
    longer silently stays exposed in public retrieval.
  - services/context.py's build_context() now fences the memories block with
    an explicit "not instructions" note before splicing it into the prompt —
    memory content is user-authored and persistent, i.e. a stored-prompt-
    injection surface.
"""

import json
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

from ..services.memory import Embedder, embedding_to_pgvector

# text in (system, user), model text out — injectable so tests never touch the
# network, same seam shape as services.memory.Embedder.
Llm = Callable[[str, str], str]

BATCH_SIZE = 20          # messages per LLM call (Group Messages sliced at 20)
DEFAULT_LOOKBACK_H = 2   # first run with no watermark: 2 hours ago (v1 default)
DEFAULT_LIMIT = 200      # rows per source per run (v1 LIMIT 200)
LLM_MODEL = "anthropic/claude-haiku-4.5"   # v1's extraction model, via OpenRouter
USER_TZ = ZoneInfo("America/New_York")     # batch-date fallback renders in owner tz

# Ported VERBATIM from the "Build Extraction Request" Code node — this prompt is
# the contract being shadow-diffed, so it must not drift from what prod runs.
EXTRACTION_SYSTEM_PROMPT = """You extract memories from conversation transcripts. Return a JSON array only -- no preamble, no explanation.

## What to EXTRACT (must have lasting value):
- Stable identity facts: name, location, job, relationships, gender, age
- Genuine interests with signal: hobbies, games, media, projects they care about
- Meaningful life events with context: trips, purchases, celebrations, milestones
- Stated preferences for Aerys: how they want to be addressed, communication style
- Vehicle, home, pet details: specific and memorable

## What to REJECT (ephemeral noise -- do NOT extract):
- Transient actions: "going to bed", "grabbing coffee", "brb"
- Session-specific decisions: "parking this bug", "let's do X next"
- Meta-commentary about the system: "the bot is broken", "that workflow failed"
- Timestamps without facts: "it's late", "been a long day"
- Greetings and smalltalk: "hey", "what's up", "how are you"

## Return format:
[{"key_label":"category.attribute","value_text":"the fact itself, naturally phrased","context":"why/how this came up in conversation (1 sentence, optional for stable facts)","event_date":"when this happened if mentioned (e.g. 'Feb 28', 'last week', null if not stated)","privacy_level":"public|private","asserted_by":"self|third_party","confidence":0.0}]

## key_label rules:
- Generic prefixes ONLY: basic, user, work, vehicle, interest, relationship, preference, event
- NEVER use a person's name as prefix
- Examples: basic.location, user.vehicle, interest.game, event.trip, preference.communication

## value_text rules:
- Write naturally, as a human would remember it -- NOT as a database entry
- BAD: "Dodge Ram"  GOOD: "Interested in a Dodge Ram truck"
- BAD: "Rotonda West"  GOOD: "Lives in Rotonda West, Florida"
- For interests, include what makes it notable: "Collects retro games, especially SNES titles"

## context rules:
- 1 sentence max, captures the conversational moment
- Include for interests, events, preferences. Skip for basic identity facts.
- BAD: "mentioned in conversation"  GOOD: "came up while joking about RAM prices"
- If the conversation is mundane, context is null

## privacy_level rules:
- "public": name, job, location, hobbies, vehicle, interests, general life facts
- "private": health details, financial specifics, relationship struggles, personal traumas, sexual orientation
- Source channel (DM vs public) is context, not a rule

If nothing worth remembering, return []."""

# Both source queries return the SAME column tuple so downstream code is
# source-agnostic. created_at comes back twice: as a datetime (for ordering /
# batch-date math) and as ::text (the RAW string the watermark stores).
SOURCE_COLUMNS = (
    "id",
    "person_id",
    "content",
    "source_platform",
    "privacy_level",
    "created_at",
    "created_at_raw",
    "speaker_name",
    "source_thread",
)

# v1 channels — the "Fetch from n8n_chat_histories" query, ported. READ-ONLY
# against prod aerys. session_id == person_id (the Core Agent's memory key).
PROD_MESSAGES_SQL = """\
WITH valid_sessions AS (
  SELECT session_id
  FROM n8n_chat_histories
  WHERE session_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
  GROUP BY session_id
)
SELECT
  h.id::text AS id,
  vs.session_id::uuid AS person_id,
  (h.message->>'content') AS content,
  'discord' AS source_platform,
  'private' AS privacy_level,
  h.created_at,
  h.created_at::text AS created_at_raw,
  COALESCE(pers.display_name, 'Unknown') AS speaker_name,
  'v1:n8n_chat_histories' AS source_thread
FROM n8n_chat_histories h
JOIN valid_sessions vs ON vs.session_id = h.session_id
LEFT JOIN persons pers ON pers.id = vs.session_id::uuid
WHERE (h.message->>'type') = 'human'
  AND h.message->>'content' IS NOT NULL
  AND h.message->>'content' != ''
  AND h.created_at > %(after)s::timestamptz
ORDER BY h.created_at ASC
LIMIT %(limit)s
"""

# Her own conversations — the v2 audit spine. Lives in the SAME aerys_v2 database
# the staging tables do. No persons table on this side (identity is prod data),
# so speaker stays 'Unknown' — the transcript prompt tolerates it. Guild turns are
# public context, everything else (DM/voice/cli) is private — same rule the v1
# adapters applied per channel.
V2_TURNS_SQL = """\
SELECT
  t.id::text AS id,
  t.person_id,
  t.input_text AS content,
  t.channel AS source_platform,
  CASE WHEN t.channel = 'guild' THEN 'public' ELSE 'private' END AS privacy_level,
  t.created_at,
  t.created_at::text AS created_at_raw,
  'Unknown' AS speaker_name,
  t.thread_id AS source_thread
FROM v2_turns t
WHERE t.person_id IS NOT NULL
  AND t.input_text != ''
  AND t.created_at > %(after)s::timestamptz
ORDER BY t.created_at ASC
LIMIT %(limit)s
"""

INSERT_STAGING_SQL = """\
INSERT INTO v2_memories_staging
  (person_id, content, key_label, context, event_date, embedding,
   source_platform, privacy_level, created_at, source_thread)
VALUES
  (%(person_id)s::uuid, %(content)s, %(key_label)s, %(context)s, %(event_date)s,
   %(embedding)s::vector, %(source_platform)s, %(privacy_level)s,
   %(created_at)s::timestamptz, %(source_thread)s)
"""

WATERMARK_GET_SQL = """\
SELECT last_processed_at FROM v2_extraction_watermark WHERE source = %(source)s
"""

WATERMARK_SET_SQL = """\
INSERT INTO v2_extraction_watermark (source, last_processed_at, updated_at)
VALUES (%(source)s, %(raw)s, now())
ON CONFLICT (source) DO UPDATE
  SET last_processed_at = EXCLUDED.last_processed_at, updated_at = now()
"""

# --- live-mode triage (prod `memories`, not staging) --------------------------

# The dedup check — the unique index's actual shape: one LIVE row per
# (person_id, key_label). ORDER BY + LIMIT 1 is belt-and-braces (the index
# should already guarantee at most one live row) rather than trusting it blind.
TRIAGE_SELECT_SQL = """\
SELECT id, content FROM memories
WHERE person_id = %(person_id)s::uuid AND key_label = %(key_label)s
  AND deleted_at IS NULL
ORDER BY created_at DESC
LIMIT 1
"""

# No existing row for this (person_id, key_label): a plain insert, same shape
# as INSERT_STAGING_SQL but into prod's table and with updated_at (staging has
# no updated_at column — append-only never needs one).
TRIAGE_INSERT_SQL = """\
INSERT INTO memories
  (person_id, content, key_label, context, event_date, embedding,
   source_platform, privacy_level, created_at, updated_at)
VALUES
  (%(person_id)s::uuid, %(content)s, %(key_label)s, %(context)s, %(event_date)s,
   %(embedding)s::vector, %(source_platform)s, %(privacy_level)s,
   %(created_at)s::timestamptz, now())
"""

# Same-value dupe: refresh content/context/event_date/embedding in place,
# updated_at=NOW() marks the refresh — created_at is deliberately NOT touched
# (it still means "when this fact first landed", the h.created_at lesson).
# privacy_level is included but only allowed to TIGHTEN (public -> private):
# the CASE means a re-statement tagged 'private' always wins (a fact re-stated
# more carefully must not stay silently public), while a re-statement tagged
# 'public' can NEVER declassify an existing 'private' row on its own — that
# direction is a real leak (a private row starts showing up in public
# retrieval) and an UPDATE branch triggered by a routine restatement is not
# the place to make that call.
TRIAGE_UPDATE_SQL = """\
UPDATE memories
SET content = %(content)s, context = %(context)s, event_date = %(event_date)s,
    embedding = %(embedding)s::vector, updated_at = now(),
    privacy_level = CASE WHEN %(privacy_level)s = 'private' THEN 'private'
                         ELSE privacy_level END
WHERE id = %(id)s
"""

# Different value: soft-delete the old row and insert the new one atomically —
# the CTE pattern from CLAUDE.md ("atomic row replacement"), eliminating the
# race a separate UPDATE-then-INSERT would open between parallel writers.
TRIAGE_REPLACE_SQL = """\
WITH soft_del AS (
  UPDATE memories SET deleted_at = now() WHERE id = %(old_id)s RETURNING id
)
INSERT INTO memories
  (person_id, content, key_label, context, event_date, embedding,
   source_platform, privacy_level, created_at, updated_at)
VALUES
  (%(person_id)s::uuid, %(content)s, %(key_label)s, %(context)s, %(event_date)s,
   %(embedding)s::vector, %(source_platform)s, %(privacy_level)s,
   %(created_at)s::timestamptz, now())
"""

# the "two armed writers" gate — kind not seeded by migration 001 (only
# emit/ha_write/governance/email exist there); live mode seeds it itself on
# first touch, same 'n8n'-holds-by-default posture as the other four kinds.
LEASE_KIND = "memory_extraction"

LEASE_SELECT_SQL = "SELECT holder FROM v2_writer_lease WHERE kind = %(kind)s"

LEASE_INSERT_IF_ABSENT_SQL = """\
INSERT INTO v2_writer_lease (kind, holder, note)
VALUES (%(kind)s, 'n8n', 'pre-flip')
ON CONFLICT (kind) DO NOTHING
"""

# The brain-vs-brain half of the two-armed-writers gate. ensure_lease_holder
# only answers "who is AUTHORIZED to write live" (n8n vs brain) — it is a
# plain SELECT, so it does nothing to stop TWO 'brain' processes (the loop
# container + a manual `--once --live` on the same box) from triaging the
# same (person_id, key_label) at once. pg_try_advisory_xact_lock is a real
# Postgres mutex: non-blocking (returns false instead of queuing forever) and
# scoped to the CURRENT transaction, so it releases itself automatically on
# commit OR rollback — a crashed pass can never leave a stuck lock behind.
LEASE_TRY_LOCK_SQL = "SELECT pg_try_advisory_xact_lock(hashtext(%(kind)s))"

# "same value restated" vs "value changed" — exact match short-circuits (the
# common case: near-identical LLM phrasing of an unchanged fact); below that,
# a case/whitespace-insensitive fuzzy ratio. Picked empirically generous
# enough that reworded-but-same facts UPDATE instead of needlessly REPLACE-ing
# (which soft-deletes real history) — a genuinely new value (new city, new
# job) scores well below this on SequenceMatcher.
VALUE_SIMILARITY_THRESHOLD = 0.85

class LiveWriteRefused(RuntimeError):
    """A live-mode hard gate tripped — refuse to write, never silently skip.

    Same posture as config.BootConfigError: a write surface aimed at a
    dangerous state (two armed writers, or a lease still held by n8n) must
    refuse loudly before touching anything, not degrade quietly.
    """


def _value_text_from_content(content: str) -> str:
    """Recover value_text from a composed content string — the inverse of
    _compose_content's `key_label: value_text -- context (event_date)`.
    Only the value matters for the triage comparison, so context/date are
    stripped, not parsed: the FIRST ': ' ends the key_label, the FIRST
    ' -- ' (if any) starts context, a trailing ' (...)' (if any) is the date.
    """
    value = content.split(": ", 1)[-1] if ": " in content else content
    value = value.split(" -- ", 1)[0]
    value = re.sub(r" \([^()]*\)$", "", value)
    return value.strip()


def values_similar(a: str, b: str, *, threshold: float = VALUE_SIMILARITY_THRESHOLD) -> bool:
    """Is `b` a restatement of `a`, or a genuinely different value?

    Case/whitespace-normalized; exact match short-circuits before the
    SequenceMatcher ratio (cheap, and the overwhelmingly common case).
    """
    a_n, b_n = a.strip().lower(), b.strip().lower()
    if a_n == b_n:
        return True
    return SequenceMatcher(None, a_n, b_n).ratio() >= threshold


def triage_memory(
    conn: Any,
    *,
    person_id: str,
    key_label: str | None,
    value_text: str,
    content: str,
    context: str | None,
    event_date: str | None,
    embedding: str,
    source_platform: str,
    privacy_level: str,
    created_at: str,
) -> str:
    """The "Insert Memory" node's real dedup branch, ported to prod `memories`.

    Returns 'insert' | 'update' | 'replace' | 'skipped' — the action actually
    taken, for the caller's per-source stats. embedding must already be a
    FRESH pgvector string (embedding_to_pgvector(embedder(content))) — live
    mode never reuses an embedding across writes, unlike a cache would.

    Two refusals BEFORE any query runs, both cross-review corruption findings:
      - key_label missing/blank: the dedup index is (person_id, key_label);
        `key_label = NULL` is never true in SQL, so a NULL key_label makes
        the dedup SELECT always miss and every observation falls to INSERT —
        unbounded duplicate rows. Unusable for dedup, so refuse outright.
      - value_text missing/blank: values_similar(existing, "") == 0.0 (an
        empty string never matches ANY real value), which forces the REPLACE
        branch — soft-deleting the real memory and inserting a blank
        "key_label: " row in its place. An observation that says nothing
        must never be allowed to overwrite one that does.
    """
    if key_label is None or not key_label.strip():
        return "skipped"
    if not value_text.strip():
        return "skipped"

    params = {
        "person_id": person_id,
        "content": content,
        "key_label": key_label,
        "context": context,
        "event_date": event_date,
        "embedding": embedding,
        "source_platform": source_platform,
        "privacy_level": privacy_level,
        "created_at": created_at,
    }

    existing = conn.execute(TRIAGE_SELECT_SQL, {"person_id": person_id, "key_label": key_label}).fetchone()
    if existing is None:
        conn.execute(TRIAGE_INSERT_SQL, params)
        return "insert"

    existing_id, existing_content = existing
    if values_similar(_value_text_from_content(existing_content), value_text):
        conn.execute(
            TRIAGE_UPDATE_SQL,
            {"id": existing_id, "content": content, "context": context,
             "event_date": event_date, "embedding": embedding,
             "privacy_level": privacy_level},
        )
        return "update"

    conn.execute(TRIAGE_REPLACE_SQL, {**params, "old_id": existing_id})
    return "replace"


def ensure_lease_holder(staging_conn: Any) -> str:
    """The writer-lease read half of the two-armed-writers gate.

    Seeds the row as holder='n8n' if this is the first live-mode touch ever
    (migration 001 only seeded the original four kinds), then returns whoever
    holds it — 'n8n' or 'brain'. NEVER flips it: the cutover is a deliberate
    separate act (a human or a flip script), not something a worker pass does
    to itself just by running.
    """
    staging_conn.execute(LEASE_INSERT_IF_ABSENT_SQL, {"kind": LEASE_KIND})
    row = staging_conn.execute(LEASE_SELECT_SQL, {"kind": LEASE_KIND}).fetchone()
    return row[0] if row else "n8n"


def acquire_write_mutex(staging_conn: Any, *, kind: str = LEASE_KIND) -> bool:
    """The brain-vs-brain half of the two-armed-writers gate — see
    LEASE_TRY_LOCK_SQL. ensure_lease_holder answers "who is authorized"; this
    answers "is anyone ELSE already writing right now". Non-blocking (a
    concurrent holder makes this return False immediately, never queues) and
    transaction-scoped (auto-released on commit/rollback — nothing to clean
    up, nothing to leak on a crash).
    """
    row = staging_conn.execute(LEASE_TRY_LOCK_SQL, {"kind": kind}).fetchone()
    return bool(row and row[0])


def read_watermark(staging_conn: Any, source: str, *, lookback_hours: int = DEFAULT_LOOKBACK_H) -> str:
    """The "Read Last Processed" node — but durable, not workflow staticData.

    Returns the stored RAW Postgres timestamp string, or (first run only) an
    ISO string for `lookback_hours` ago. The default is the ONE place a
    Python-formatted timestamp is allowed: it never came from the database, so
    there is no raw string to preserve and no row it could falsely re-match.
    """
    row = staging_conn.execute(WATERMARK_GET_SQL, {"source": source}).fetchone()
    if row and row[0]:
        return row[0]
    return (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()


def save_watermark(staging_conn: Any, source: str, raw: str) -> None:
    """Persist the raw string VERBATIM — see the module docstring's precision bug."""
    staging_conn.execute(WATERMARK_SET_SQL, {"source": source, "raw": raw})


def group_by_person(rows: list[dict], *, batch_size: int = BATCH_SIZE) -> list[dict]:
    """The "Group Messages" Code node: person_id FIRST, then slice into batches.

    Grouping before extraction is load-bearing (pre-05.1 lesson): one mixed
    transcript per LLM call meant every fact landed under every participant.
    Rows with a NULL person_id are dropped, same as v1's guard node.
    """
    by_person: dict[str, list[dict]] = {}
    for row in rows:
        pid = row.get("person_id")
        if not pid:
            continue
        by_person.setdefault(str(pid), []).append(row)

    groups = []
    for pid, msgs in by_person.items():
        for i in range(0, len(msgs), batch_size):
            batch = msgs[i : i + batch_size]
            # latest message in the batch: memory created_at + batch-date fallback
            latest = max(batch, key=lambda m: m["created_at"])
            groups.append(
                {
                    "person_id": pid,
                    "messages": batch,
                    "source_platform": batch[0].get("source_platform") or "discord",
                    "privacy_level": batch[0].get("privacy_level") or "public",
                    "source_thread": batch[0].get("source_thread") or "unknown",
                    "latest": latest,
                }
            )
    return groups


def build_transcript(messages: list[dict]) -> str:
    """The transcript half of "Build Extraction Request": `[Speaker]: text` lines."""
    return "\n".join(f"[{m.get('speaker_name') or 'Unknown'}]: {m['content']}" for m in messages)


def batch_date(latest_created_at: datetime) -> str:
    """The "Parse Observations" fallback event_date: 'Jul 2' in the owner's tz."""
    local = latest_created_at.astimezone(USER_TZ)
    return f"{local.strftime('%b')} {local.day}"   # no %-d: portable day-without-zero


def parse_observations(text: str) -> list[dict] | None:
    """The parse half of "Parse Observations": strip code fences, tolerate garbage.

    Returns [] for a GENUINE empty extraction (the model looked and found
    nothing worth remembering — EXTRACTION_SYSTEM_PROMPT's documented `[]`
    reply) — the caller must still advance the watermark past these rows, or
    they get reprocessed forever. Returns None when the text could not be
    read as a JSON array AT ALL: max_tokens=1200 makes truncation routine on
    a busy transcript, and a truncated reply is NOT "no memories" — it's a
    parse failure. The two must never be conflated: the caller must NEVER
    advance the watermark past a group whose reply parsed to None, or those
    messages are lost forever (the watermark only ever moves forward).
    A non-list top-level value (an object, a bare string, `null`...) is
    ALSO treated as a failure, not an empty — the prompt's contract is "JSON
    array only", so anything else is the model failing to comply, not a
    considered "nothing here" answer, and the SAME lost-forever risk applies
    if we treated it as an empty. Garbage items *inside* an otherwise-valid
    array (non-dict entries) are still dropped silently — the array parsed
    fine, so that's noise, not a batch failure.
    """
    cleaned = re.sub(r"^```(?:json)?\n?", "", text.strip(), flags=re.M)
    cleaned = re.sub(r"\n?```$", "", cleaned, flags=re.M).strip()
    try:
        observations = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(observations, list):
        return None
    return [o for o in observations if isinstance(o, dict)]


def _compose_content(obs: dict, fallback_date: str | None) -> tuple[str, str | None]:
    """content string + effective event_date — the composition rules of v1's
    Parse Observations node: `key_label: value_text -- context (event_date)`."""
    event_date = obs.get("event_date") or fallback_date
    content = f"{obs.get('key_label') or ''}: {obs.get('value_text') or ''}"
    if obs.get("context"):
        content += f" -- {obs['context']}"
    if event_date:
        content += f" ({event_date})"
    return content, event_date


def _trim_tie_boundary(rows: list[dict], limit: int) -> list[dict]:
    """The LIMIT-boundary keyset-pagination hazard: both source queries are
    `ORDER BY created_at ASC LIMIT %(limit)s`, and the watermark advances to
    max(created_at) of whatever came back. If the true backlog is >limit rows
    AND the row at the cut line ties (identical created_at) with the row
    before it — real, since `now()` is transaction-time and a batch insert
    can share one timestamp — the tied sibling(s) just past the cut are
    invisible to this pass. Advancing the watermark to the tie's timestamp
    would then permanently skip them (`created_at > watermark` excludes
    exact ties on every future pass too).

    Callers fetch `limit + 1` (one past what they actually want) so this can
    detect "there may be more beyond the cut": if fewer than `limit + 1` rows
    came back, the fetch wasn't full — there's nothing beyond it, no tie risk,
    return unchanged. If it WAS full, trim the entire trailing run of rows
    that share the last row's created_at, so the watermark can only ever
    land on a timestamp with none of its siblings left un-fetched — the
    whole tied group arrives together, next pass, instead of being split.

    Pathological fallback: if EVERY fetched row shares one timestamp (the
    whole `limit + 1` overfetch is one giant tie), there is nothing safe to
    trim to — return the rows unchanged rather than stall forever on a
    situation that isn't realistically reachable at microsecond precision.
    """
    if len(rows) <= limit:
        return rows
    boundary_ts = rows[-1]["created_at"]
    cutoff = len(rows)
    while cutoff > 0 and rows[cutoff - 1]["created_at"] == boundary_ts:
        cutoff -= 1
    return rows[:cutoff] if cutoff else rows


def _safe_watermark(rows: list[dict], failed_row_ids: set[str], after: str) -> str:
    """The watermark to persist for one source's pass.

    No failures: unchanged behavior — advance to the newest row's raw
    created_at (string round-trip preserved, never a reformatted datetime).

    One or more groups failed to parse (parse_observations returned None,
    not a genuine []): advancing past those messages would lose them
    forever, since the watermark only ever moves forward and nothing else
    ever re-reads what's behind it. So freeze the mark at the newest row
    STRICTLY OLDER than the earliest failed row — every row at or after
    that point (failed or not, tied or not) is left unadvanced and gets
    swept up and retried together on the next pass. If the very first row
    already failed, there is nothing older to advance to: `after` (the
    pass's own starting point) is returned unchanged, and the caller still
    re-persists it explicitly — a repeatedly-failing first-ever run must not
    let the lookback-hours fallback silently drift forward with wall-clock
    time (read_watermark recomputes "now - lookback" fresh every time there's
    no stored row).
    """
    if not failed_row_ids:
        newest = max(rows, key=lambda r: r["created_at"])
        return newest["created_at_raw"]
    failed_ts = min(r["created_at"] for r in rows if r["id"] in failed_row_ids)
    older = [r for r in rows if r["created_at"] < failed_ts]
    if not older:
        return after
    return max(older, key=lambda r: r["created_at"])["created_at_raw"]


def run_extraction(
    source_conn: Any,
    staging_conn: Any,
    llm: Llm,
    embedder: Embedder,
    *,
    lookback_hours: int = DEFAULT_LOOKBACK_H,
    batch_limit: int = DEFAULT_LIMIT,
) -> dict:
    """One shadow extraction pass over both conversation sources.

    source_conn  — prod aerys, READ-ONLY (n8n_chat_histories + persons SELECTs only)
    staging_conn — aerys_v2: v2_turns reads + ALL writes (staging + watermark)

    Per source: watermark -> fetch -> group by person -> LLM -> embed -> stage ->
    advance watermark (only after the batch fully landed, so a crash re-runs it —
    the staging insert is append-only, so a re-run at worst duplicates shadow rows,
    never loses them). Returns a per-source summary dict for logs/reports.

    Fetches `batch_limit + 1` rows and hands them to _trim_tie_boundary — the
    "+1" is how that helper detects whether the fetch actually hit the limit
    (more rows may exist beyond it) versus legitimately exhausting the
    backlog; see its docstring for the LIMIT-boundary tie hazard this closes.

    A group whose reply fails to parse (parse_observations returns None, not
    a genuine []) contributes ZERO inserts and is never allowed to pull the
    watermark past its own messages — see _safe_watermark. It's still safe
    to advance past every OTHER row in the batch: a re-run only ever risks a
    harmless duplicate shadow insert, never a lost one.
    """
    sources = (
        ("prod_chat", source_conn, PROD_MESSAGES_SQL),
        ("v2_turns", staging_conn, V2_TURNS_SQL),
    )
    summary: dict[str, Any] = {"sources": {}, "inserted_total": 0}

    for name, conn, sql in sources:
        after = read_watermark(staging_conn, name, lookback_hours=lookback_hours)
        raw_rows = conn.execute(sql, {"after": after, "limit": batch_limit + 1}).fetchall()
        rows = [dict(zip(SOURCE_COLUMNS, r)) for r in raw_rows]
        rows = _trim_tie_boundary(rows, batch_limit)

        stats = {
            "rows": len(rows), "groups": 0, "observations": 0, "inserted": 0,
            "parse_failures": 0, "watermark": None,
        }
        summary["sources"][name] = stats
        if not rows:
            continue  # empty window: no LLM spend, watermark untouched (v1 behavior)

        groups = group_by_person(rows)
        stats["groups"] = len(groups)
        failed_row_ids: set[str] = set()

        for group in groups:
            reply = llm(
                EXTRACTION_SYSTEM_PROMPT,
                f"Extract observations from this conversation:\n\n{build_transcript(group['messages'])}",
            )
            observations = parse_observations(reply)
            if observations is None:
                # Parse failure, not "nothing here" — never stage anything for
                # this group, and never let the watermark advance past it.
                stats["parse_failures"] += 1
                failed_row_ids.update(m["id"] for m in group["messages"])
                continue
            stats["observations"] += len(observations)
            fallback = batch_date(group["latest"]["created_at"])

            for obs in observations:
                content, event_date = _compose_content(obs, fallback)
                staging_conn.execute(
                    INSERT_STAGING_SQL,
                    {
                        "person_id": group["person_id"],
                        "content": content,
                        "key_label": obs.get("key_label"),
                        "context": obs.get("context"),
                        "event_date": event_date,
                        "embedding": embedding_to_pgvector(embedder(content)),
                        "source_platform": group["source_platform"],
                        "privacy_level": obs.get("privacy_level") or group["privacy_level"],
                        # original message time, not now() — the h.created_at lesson
                        "created_at": group["latest"]["created_at_raw"],
                        "source_thread": group["source_thread"],
                    },
                )
                stats["inserted"] += 1
                summary["inserted_total"] += 1

        watermark_raw = _safe_watermark(rows, failed_row_ids, after)
        save_watermark(staging_conn, name, watermark_raw)
        stats["watermark"] = watermark_raw

    return summary


_TRIAGE_STAT_KEYS = {
    "insert": "inserted", "update": "updated", "replace": "replaced", "skipped": "skipped",
}


def run_live_extraction(
    source_conn: Any,
    staging_conn: Any,
    prod_write_conn: Any,
    llm: Llm,
    embedder: Embedder,
    *,
    lookback_hours: int = DEFAULT_LOOKBACK_H,
    batch_limit: int = DEFAULT_LIMIT,
) -> dict:
    """The live counterpart of run_extraction(): same read -> group -> LLM ->
    embed pipeline, same watermark table (both modes share the high-water
    mark — a shadow pass and a live pass must never re-extract each other's
    already-processed rows), but observations land in prod `memories` via
    triage_memory() instead of the append-only v2_memories_staging.

    source_conn      — prod aerys, READ-ONLY (unchanged from shadow mode)
    staging_conn      — aerys_v2: v2_turns reads, watermark, writer-lease
    prod_write_conn   — A THIRD connection: prod aerys again, but READ-WRITE
                        this time (a separate connection object from
                        source_conn — you cannot write on a read_only=True
                        session, so the same database gets two connections
                        with two different postures)

    Hard gates run FIRST, before any read or LLM spend — refuse loudly and
    early, run_boot_assertions' stance applied to a single pass:
      1. v2_writer_lease[kind='memory_extraction'] must be held by 'brain'.
      2. This pass must actually ACQUIRE the write mutex (acquire_write_mutex)
         — (1) only says brain is AUTHORIZED; a second 'brain' process (the
         loop container racing a manual `--once --live`) would pass (1) just
         as easily. This is the real concurrency control.
    Both raise LiveWriteRefused, never a silent skip. (An n8n-liveness gate used
    to precede these — removed 2026-07-05 when n8n was retired; the write-lease is
    the sole authorization guard now.)

    Fetch/trim/watermark-safety are identical to run_extraction (same
    _trim_tie_boundary / _safe_watermark helpers, same shared watermark
    table) — see run_extraction's docstring for the LIMIT-tie and
    parse-failure hazards those close. A group whose reply fails to parse
    here triages ZERO observations and never pulls the watermark past its
    own messages, exactly as in shadow mode.
    """
    holder = ensure_lease_holder(staging_conn)
    if holder != "brain":
        raise LiveWriteRefused(
            f"v2_writer_lease[{LEASE_KIND!r}] is held by {holder!r}, not 'brain' — "
            "refusing to write to prod memories until the lease flips."
        )
    if not acquire_write_mutex(staging_conn):
        raise LiveWriteRefused(
            f"could not acquire the {LEASE_KIND!r} write mutex — another live "
            "extraction pass is already running; refusing to run two writers "
            "concurrently against the same (person_id, key_label) rows."
        )

    sources = (
        ("prod_chat", source_conn, PROD_MESSAGES_SQL),
        ("v2_turns", staging_conn, V2_TURNS_SQL),
    )
    summary: dict[str, Any] = {
        "sources": {},
        "inserted_total": 0,
        "updated_total": 0,
        "replaced_total": 0,
        "skipped_total": 0,
    }

    for name, conn, sql in sources:
        after = read_watermark(staging_conn, name, lookback_hours=lookback_hours)
        raw_rows = conn.execute(sql, {"after": after, "limit": batch_limit + 1}).fetchall()
        rows = [dict(zip(SOURCE_COLUMNS, r)) for r in raw_rows]
        rows = _trim_tie_boundary(rows, batch_limit)

        stats = {
            "rows": len(rows), "groups": 0, "observations": 0,
            "inserted": 0, "updated": 0, "replaced": 0, "skipped": 0,
            "parse_failures": 0, "watermark": None,
        }
        summary["sources"][name] = stats
        if not rows:
            continue  # empty window: no LLM spend, watermark untouched (v1 behavior)

        groups = group_by_person(rows)
        stats["groups"] = len(groups)
        failed_row_ids: set[str] = set()

        for group in groups:
            reply = llm(
                EXTRACTION_SYSTEM_PROMPT,
                f"Extract observations from this conversation:\n\n{build_transcript(group['messages'])}",
            )
            observations = parse_observations(reply)
            if observations is None:
                # Parse failure, not "nothing here" — never triage anything for
                # this group, and never let the watermark advance past it.
                stats["parse_failures"] += 1
                failed_row_ids.update(m["id"] for m in group["messages"])
                continue
            stats["observations"] += len(observations)
            fallback = batch_date(group["latest"]["created_at"])

            for obs in observations:
                content, event_date = _compose_content(obs, fallback)
                # embedding computed FRESH every write — never cached/reused,
                # so an UPDATEd memory's embedding never silently goes stale.
                action = triage_memory(
                    prod_write_conn,
                    person_id=group["person_id"],
                    key_label=obs.get("key_label"),
                    value_text=obs.get("value_text") or "",
                    content=content,
                    context=obs.get("context"),
                    event_date=event_date,
                    embedding=embedding_to_pgvector(embedder(content)),
                    source_platform=group["source_platform"],
                    privacy_level=obs.get("privacy_level") or group["privacy_level"],
                    # original message time, not now() — the h.created_at lesson
                    created_at=group["latest"]["created_at_raw"],
                )
                stat_key = _TRIAGE_STAT_KEYS[action]
                stats[stat_key] += 1
                summary[f"{stat_key}_total"] += 1

        watermark_raw = _safe_watermark(rows, failed_row_ids, after)
        save_watermark(staging_conn, name, watermark_raw)
        stats["watermark"] = watermark_raw

    return summary


def openrouter_chat(api_key: str, *, model: str = LLM_MODEL,
                    base_url: str = "https://openrouter.ai/api/v1",
                    timeout_s: float = 120.0) -> Llm:
    """The real Llm seam — the "Call LLM for Extraction" HTTP Request node.

    Same OpenRouter host/key the embedder uses (settings.embeddings_api_key IS an
    OpenRouter key); stdlib urllib for the same dependency-free reason as
    services.memory.openrouter_embedder. temperature/max_tokens match v1 exactly.
    """

    def llm(system: str, user: str) -> str:
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(
                {
                    "model": model,
                    "temperature": 0.1,
                    "max_tokens": 1200,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
            ).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            payload = json.load(resp)
        return payload["choices"][0]["message"]["content"] or "[]"

    return llm
