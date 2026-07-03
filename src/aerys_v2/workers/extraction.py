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
"""

import json
import re
import urllib.request
from datetime import datetime, timedelta, timezone
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


def parse_observations(text: str) -> list[dict]:
    """The parse half of "Parse Observations": strip code fences, tolerate garbage.

    Any parse failure or non-array result is an empty extraction, never a crash —
    a chatty model must not take the whole batch run down.
    """
    cleaned = re.sub(r"^```(?:json)?\n?", "", text.strip(), flags=re.M)
    cleaned = re.sub(r"\n?```$", "", cleaned, flags=re.M).strip()
    try:
        observations = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(observations, list):
        return []
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
    """
    sources = (
        ("prod_chat", source_conn, PROD_MESSAGES_SQL),
        ("v2_turns", staging_conn, V2_TURNS_SQL),
    )
    summary: dict[str, Any] = {"sources": {}, "inserted_total": 0}

    for name, conn, sql in sources:
        after = read_watermark(staging_conn, name, lookback_hours=lookback_hours)
        raw_rows = conn.execute(sql, {"after": after, "limit": batch_limit}).fetchall()
        rows = [dict(zip(SOURCE_COLUMNS, r)) for r in raw_rows]

        stats = {"rows": len(rows), "groups": 0, "observations": 0, "inserted": 0, "watermark": None}
        summary["sources"][name] = stats
        if not rows:
            continue  # empty window: no LLM spend, watermark untouched (v1 behavior)

        groups = group_by_person(rows)
        stats["groups"] = len(groups)

        for group in groups:
            reply = llm(
                EXTRACTION_SYSTEM_PROMPT,
                f"Extract observations from this conversation:\n\n{build_transcript(group['messages'])}",
            )
            observations = parse_observations(reply)
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

        # max by the REAL timestamp, store its RAW string — string-max would
        # misorder Postgres's variable-width fractional seconds ('.9' > '.15').
        newest = max(rows, key=lambda r: r["created_at"])
        save_watermark(staging_conn, name, newest["created_at_raw"])
        stats["watermark"] = newest["created_at_raw"]

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
