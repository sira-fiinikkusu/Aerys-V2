"""Memory retrieval — n8n 04-02 Memory Retrieval (GXaRTmTCTP9XqQxY) as plain functions.

READ-ONLY MODULE: SELECT only.

n8n quirk-workarounds killed:
  - `UNION ALL SELECT NULL::uuid, ... 0.0 WHERE NOT EXISTS(...)` zero-row sentinel —
    n8n drops items on 0 rows; Python returns []. The sentinel-row filter in the
    old Format node (`content is None`) is kept as a cheap belt-and-braces check.
  - TWO whole query variants (private/public differing only in the privacy_level
    predicate, duplicated again inside the sentinel's NOT EXISTS) — collapsed to one
    query with a `= ANY(%(levels)s)` list param.

Latent V1 bug FIXED in the port: rows with NULL `m.embedding` produced NULL
combined_score, and `ORDER BY ... DESC` in Postgres is NULLS FIRST — so memories
that were never embedded sorted to the TOP of retrieval. `AND m.embedding IS NOT
NULL` excludes them (an unembedded memory can't be similarity-scored anyway).

Scoring (in SQL, reference implementation in `combined_score()` below):
  score = cosine_similarity * (0.7 + 0.3 * recency)      -- recency is a BOOST, not a term
  cosine_similarity = 1 - (embedding <=> query_vec)      -- pgvector cosine distance
  recency = clamp(1 - age_seconds / 30 days, 0, 1)       -- linear decay to 0 at 30d
No similarity threshold: ALL of a person's live memories are scored, top RETRIEVE_LIMIT (30) kept.

V1 scoring bug FIXED in the port (observed live 2026-07-03): the additive form
`sim*0.7 + recency*0.3` let ANY fresh memory (+0.3) outrank every memory older
than 30 days regardless of relevance — "who am I married to?" retrieved this
week's smart-home chatter while the April wedding memories sat at rank #21+,
below the LIMIT. Multiplicative recency keeps the freshness preference but makes
similarity the gate: an irrelevant memory can't ride recency to the top, and an
old-but-on-point memory (sim*0.7 floor) beats fresh noise.
"""

import json
import math
import re
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

# The scoring constants — named here, interpolated into the SQL below so the
# query and the reference math can never drift apart. SIM_WEIGHT is the floor
# multiplier for a fully-aged memory; a fresh one is boosted up to
# SIM_WEIGHT + RECENCY_WEIGHT (= 1.0, pure similarity).
SIM_WEIGHT = 0.7
RECENCY_WEIGHT = 0.3
RECENCY_WINDOW_S = 2_592_000  # 30 days — recency decays linearly to 0 over this
RETRIEVE_LIMIT = 30  # rows scored & returned by SQL (2x CONTEXT_CAP: room for the dedup to drop rows)
# Chris 11/25 23:23: "im good with her saving more often, though it would reinforce the need to do 15 lines instead
# of 5" (D107 keeps ~4x more tastes; on the s100 memory test 15 lines scored the same as 5: 42 vs 40 recall-only,
# 81 vs 82 with search). 5 dated from the small-context days.
CONTEXT_CAP = 15  # lines that survive dedup into the prompt block
#: How recall marks a line of HER own words (D97): what she said then, never a checked fact and
#: never something he said. services/context.py adds the matching note to the block header.
HER_WORDS_LABEL = "(her own words, not a checked fact)"
#: D105 (Chris 11/25 20:27, after "that isnt how anyone remembers anything" and "im not sure how we got on the
#: whole path that her memories of everything is funneled as spoken by me"): every recalled line says where it
#: came from. Unmarked = drawn from what the person said (extracted from their own messages, or kept by the
#: remember tool as a quote of their words). Each mark is set from what the row already carries, never guessed.
OWN_NOTE_LABEL = "(her own note)"
KAEL_NOTES_LABEL = "(from Kael's notes, approved by them)"
KAEL_TALK_LABEL = "(her talk with Kael)"
HER_PHRASING_LABEL = "(saved in her words)"
#: Codex D105 r1: reading something before saving a fact isn't proof the fact came from it.
READ_LABEL = "(saved after she read: {what})"
#: The contexts her own writers stamp (the home body's gap reports, run notes and skill reviews).
_OWN_NOTE_CONTEXTS = ("portable gap report", "note to self from unattended run", "skill review mark", "skill:")
#: Code-owned outside-source names (outside_guard) in plain words.
_READ_WHAT = {"read_web_page": "a web page", "web_search": "a web search", "read_email": "an email",
              "search_email": "an email", "room": "a group chat", "relay": "a relayed message",
              "read_file": "a file", "browse": "a web page"}
#: What each mark means, said once under the lines (only for the marks shown).
SOURCE_NOTES = {
    "hers": (f"Lines marked {HER_WORDS_LABEL} are what you (Aerys) said at the time, not verified facts and not "
             "things he said: if one matters, check it before repeating it as settled."),
    "own_note": (f"Lines marked {OWN_NOTE_LABEL} are notes you (Aerys) wrote for yourself (gap reports, run "
                 "notes, skill reviews, session summaries): yours, not things anyone told you."),
    "kael_talk": (f"Lines marked {KAEL_TALK_LABEL} are from your (Aerys) conversations with Kael, not things "
                  "they said."),
    "kael": (f"Lines marked {KAEL_NOTES_LABEL} came from Kael's notes on their records, which they approved, "
             "not from your conversations with them."),
    "her_phrasing": (f"Lines marked {HER_PHRASING_LABEL} are facts you saved in your own wording: many restate "
                     "what they told you, some are what you worked out. Say \"I have a note that...\", never quote "
                     "one as their exact words."),
    "read": ("Lines marked (saved after she read: ...) were saved right after you read that: the fact may come "
             "from it or from them. Don't present one as their words, and don't deny they said it either; if it "
             "matters, say where you think it came from."),
}


def source_mark(row: dict, key: str = "") -> tuple[str, str]:
    """(kind, label) for where a memory came from, from what the row carries; ("", "") = their words."""
    key = str(row.get("key_label") or key or "").strip().lower()
    if key.startswith("aerys."):
        return "hers", HER_WORDS_LABEL
    context = str(row.get("context") or "").strip().lower()
    # Replay 11/25: per-message recall (unlike search) can bring back a talk with Kael (a2a memory).
    if str(row.get("source_platform") or "") == "kael_line" or context.startswith("a2a exchange"):
        return "kael_talk", KAEL_TALK_LABEL
    category = [str(t) for t in (row.get("category") or ())]
    if "trust:owner" in category:
        return "", ""
    read = [t[5:] for t in category if t.startswith("from:")]
    if read:
        what = ", ".join(dict.fromkeys(_READ_WHAT.get(r, "an outside source") for r in read))
        return "read", READ_LABEL.format(what=what)
    if context.startswith(_OWN_NOTE_CONTEXTS) or key.startswith("session."):
        return "own_note", OWN_NOTE_LABEL
    if context.startswith("from kael's"):
        return "kael", KAEL_NOTES_LABEL
    if "trust:assistant" in category:
        return "her_phrasing", HER_PHRASING_LABEL
    return "", ""


def source_notes(marks) -> str:
    """The explanation lines for the marks shown, in a fixed order."""
    return "\n".join(SOURCE_NOTES[k] for k in SOURCE_NOTES if k in marks)
#: Keys whose rows add up instead of replacing (workers/extraction.APPEND_PREFIXES; a test keeps them equal).
_APPEND_PREFIXES = ("event.", "aerys.")
EMBED_MODEL = "openai/text-embedding-3-small"  # via OpenRouter, 1536-dim

# An embedder is just "text in, vector out" — injectable so tests never touch the
# network and V2 can swap providers without touching retrieval logic.
Embedder = Callable[[str], Sequence[float]]

MEMORY_SQL = f"""\
SELECT
  m.id, m.person_id, m.content, m.source_platform, m.privacy_level, m.created_at,
  (1 - (m.embedding <=> %(embedding)s::vector))
  * ({SIM_WEIGHT} + {RECENCY_WEIGHT} * LEAST(1.0, GREATEST(0.0,
      1 - EXTRACT(EPOCH FROM (NOW() - m.created_at)) / {RECENCY_WINDOW_S}.0
    ))) AS combined_score,
  m.key_label, m.context, m.category
FROM memories m
WHERE m.person_id = %(person_id)s::uuid
  AND m.deleted_at IS NULL
  AND m.embedding IS NOT NULL
  AND m.privacy_level = ANY(%(levels)s)
ORDER BY combined_score DESC
LIMIT {RETRIEVE_LIMIT}
"""

_COLUMNS = (
    "id",
    "person_id",
    "content",
    "source_platform",
    "privacy_level",
    "created_at",
    "combined_score",
    "key_label",
    "context",
    "category",
)


def combined_score(cosine_similarity: float, age_seconds: float) -> float:
    """Reference implementation of the SQL scoring math — kept for tests and parity.

    The database computes this; this function exists so the weights and the clamp
    behavior are provable offline (and eyeball-checkable against live rows).
    """
    recency = min(1.0, max(0.0, 1 - age_seconds / RECENCY_WINDOW_S))
    return cosine_similarity * (SIM_WEIGHT + RECENCY_WEIGHT * recency)


def embedding_to_pgvector(embedding: Sequence[float]) -> str:
    """Serialize a vector for pgvector's text input format: '[0.1,0.2,...]'.

    n8n mapping: the Code node did `'[' + embedding.join(',') + ']'`. Same string,
    cast with `::vector` inside the query.
    """
    return "[" + ",".join(str(v) for v in embedding) + "]"


def retrieve_memories(
    conn: Any,
    person_id: str,
    *,
    query_embedding: Sequence[float] | None = None,
    query_text: str | None = None,
    embed: Embedder | None = None,
    privacy_context: str = "public",
) -> list[dict]:
    """Score and fetch a person's top memories for a query. Returns row dicts.

    Pass either a precomputed `query_embedding`, or `query_text` + an `embed`
    seam (the n8n version always called OpenRouter inline; here the call is
    injectable — see `openrouter_embedder()` for the real one).

    Privacy branch (matches the two n8n query variants exactly): private context
    sees privacy_level in ('public','private'); public sees 'public' only.

    n8n behavior change: a missing/failed embedding fell through to `'[]'` in n8n,
    which would ERROR in Postgres — here an empty embedding short-circuits to []
    (empty context) without touching the database. Note the n8n flow embedded
    even an empty message as `''`; we preserve that (embed is called with '').
    """
    if query_embedding is None:
        if embed is None:
            raise ValueError("retrieve_memories needs query_embedding or an embed seam")
        query_embedding = embed(query_text or "")
    if not query_embedding:
        return []

    levels = ["public", "private"] if privacy_context == "private" else ["public"]
    rows = conn.execute(
        MEMORY_SQL,
        {
            "embedding": embedding_to_pgvector(query_embedding),
            "person_id": person_id,
            "levels": levels,
        },
    ).fetchall()
    return [dict(zip(_COLUMNS, row)) for row in rows]


# --- D95-A (Chris 11/24): her own search of long-term memory -------------------------------
# Recall hands every turn CONTEXT_CAP lines scored against the current message; a LongMemEval
# pilot the same day showed she could not reach anything past them. search_memories is the
# same scoring over the same table, plus a plain word match (a name the vectors rank low is
# still found), returning more rows. WHO and WHICH LEVELS are decided by the caller's scope
# (services/recall_search.py) and passed in; nothing here widens on its own. Never her
# Kael-line logs (a2a_*, as in the people block) or her skills (as on the stick).
SEARCH_LIMIT = 25  # more than CONTEXT_CAP, so a search reaches past the lines already recalled
SEARCH_POOL = 30  # rows each half fetches before the merge

_SEARCH_FROM = """\
FROM memories m
LEFT JOIN persons p ON p.id = m.person_id"""

_SEARCH_WHERE = """\
WHERE m.person_id = ANY(%(person_ids)s::uuid[])
  AND m.deleted_at IS NULL
  AND m.privacy_level = ANY(%(levels)s)
  AND (m.key_label IS NULL OR (m.key_label NOT LIKE 'a2a\\_%%' AND m.key_label NOT LIKE 'skill.%%'))
  AND m.source_platform IS DISTINCT FROM 'kael_line'"""

_SEARCH_COLUMNS = (
    "id", "person_id", "content", "key_label", "source_platform", "privacy_level",
    "created_at", "score", "display_name", "context", "category",
)

SEARCH_SEMANTIC_SQL = f"""\
SELECT
  m.id::text, m.person_id::text, m.content, m.key_label, m.source_platform, m.privacy_level,
  m.created_at,
  (1 - (m.embedding <=> %(embedding)s::vector))
  * ({SIM_WEIGHT} + {RECENCY_WEIGHT} * LEAST(1.0, GREATEST(0.0,
      1 - EXTRACT(EPOCH FROM (NOW() - m.created_at)) / {RECENCY_WINDOW_S}.0
    ))) AS score,
  p.display_name, m.context, m.category
{_SEARCH_FROM}
{_SEARCH_WHERE}
  AND m.embedding IS NOT NULL
ORDER BY score DESC
LIMIT %(limit)s
"""

# held = how many of the query's words the row holds (its text, its key, or, for someone
# other than the caller, their name: "what do you know about Sam" finds Sam's own facts).
SEARCH_KEYWORD_SQL = f"""\
SELECT
  m.id::text, m.person_id::text, m.content, m.key_label, m.source_platform, m.privacy_level,
  m.created_at, h.held AS score, p.display_name, m.context, m.category
{_SEARCH_FROM}
CROSS JOIN LATERAL (
  SELECT count(*) AS held FROM unnest(%(patterns)s::text[]) AS w(pat)
  WHERE m.content ~* w.pat OR COALESCE(m.key_label, '') ~* w.pat
     OR (m.person_id <> %(caller)s::uuid AND COALESCE(p.display_name, '') ~* w.pat)
) h
{_SEARCH_WHERE}
  AND h.held > 0
ORDER BY h.held DESC, m.created_at DESC
LIMIT %(limit)s
"""

_SEARCH_STOP = frozenset(
    "the and for with what who where when which about tell know does did you your have has "
    "from that this are was were any some said say told ever".split())


def search_words(query: str) -> list[str]:
    """The query's words worth matching on their own: 3+ letters, not filler, in order, once."""
    out: list[str] = []
    for word in re.findall(r"[^\W_]{3,}", (query or "").lower()):
        if word not in _SEARCH_STOP and word not in out:
            out.append(word)
    return out[:8]


def search_memories(
    conn: Any,
    person_ids: Sequence[str],
    levels: Sequence[str],
    query: str,
    *,
    caller: str,
    embed: Embedder | None = None,
    query_embedding: Sequence[float] | None = None,
    limit: int = SEARCH_LIMIT,
) -> list[dict]:
    """Rows holding EVERY query word first, then the vector ranking, then rows holding some
    of the words (the stick's order, so both bodies search alike). Deduplicated by id.

    Pass a precomputed `query_embedding` (so no connection is held open across the embed
    call) or an `embed` seam. A failed or missing embedding leaves the word match standing;
    it never raises for that.
    """
    query = (query or "").strip()
    if conn is None or not query or not person_ids or not levels:
        return []
    base = {"person_ids": list(person_ids), "levels": list(levels), "caller": caller,
            "limit": SEARCH_POOL}
    words = search_words(query)
    keyword: list[dict] = []
    if words:
        from aerys_v2.services.people_memory import postgres_word

        rows = conn.execute(SEARCH_KEYWORD_SQL,
                            {**base, "patterns": [postgres_word(w) for w in words]}).fetchall()
        keyword = [dict(zip(_SEARCH_COLUMNS, row)) for row in rows]
    ranked: list[dict] = []
    vector = query_embedding
    if vector is None and embed is not None:
        try:
            vector = embed(query)
        except Exception:
            vector = None
    if vector:
        rows = conn.execute(SEARCH_SEMANTIC_SQL,
                            {**base, "embedding": embedding_to_pgvector(vector)}).fetchall()
        ranked = [dict(zip(_SEARCH_COLUMNS, row)) for row in rows]
    complete = [r for r in keyword if words and int(r.get("score") or 0) >= len(words)]
    out: list[dict] = []
    seen: set[str] = set()
    for row in (*complete, *ranked, *keyword):
        key = str(row.get("id"))
        if key in seen or row.get("content") is None:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def format_memory_context(rows: list[dict], *, now: datetime | None = None) -> str:
    """The prompt-ready memory block (see format_memory_context_flagged for the pipeline)."""
    return format_memory_context_flagged(rows, now=now)[0]


def format_memory_context_flagged(rows: list[dict], *, now: datetime | None = None) -> tuple[str, bool]:
    """(block, her_shown): format_memory_context_marked's block and whether her own words (D97) are in it."""
    block, marks = format_memory_context_marked(rows, now=now)
    return block, "hers" in marks


def format_memory_context_marked(rows: list[dict], *, now: datetime | None = None) -> tuple[str, frozenset]:
    """Build the prompt-ready memory block — the "Format Memory Context" Code node.

    Pipeline preserved exactly:
      1. drop rows with no content (was the sentinel-row filter; kept as safety)
      2. dedup by key: text before the first ':' (stripped, lowered) when a colon
         exists, else the whole content — first occurrence wins, and rows arrive
         score-sorted DESC so the winner is the best-scored duplicate
      3. cap at CONTEXT_CAP (15) lines
      4. per line: display = text after the FIRST ':' (rest rejoined, so values
         containing colons — URLs, timestamps — survive), else full content;
         then '[source_platform]' if present, then '(YYYY-MM-DD, Nd ago)'
    Returns ('', False) for zero rows (caller injects nothing into the prompt); the flag is
    True only when an aerys.* row (her own words, D97) made it into the block.

    Date added vs the n8n Format node (which emitted only '(Nd ago)'): "when
    did X happen?" questions need the calendar date, and the model can't
    recover it from a bare day-count without knowing today's date.

    Cosmetic fix vs n8n: the JS template `* ${display} ${src} ${age}` left an
    interior double space when src was missing; we join parts with single spaces.
    """
    now = now or datetime.now(timezone.utc)
    seen: set[str] = set()
    lines: list[str] = []
    marks: set[str] = set()   # D97/D105: where the shown lines came from (by key and stored tags, never by text)
    for row in rows:
        content = row.get("content")
        if content is None:
            continue
        key = (
            content.split(":")[0].strip().lower()
            if ":" in content
            else content.strip().lower()
        )
        # D97: an event, an interest or her own words may hold several distinct rows
        # under one key; each is its own memory (dedup those on the whole line).
        dedup = content.strip().lower() if key.startswith(_APPEND_PREFIXES) else key
        if dedup in seen:
            continue
        seen.add(dedup)

        if ":" in content:
            display = ":".join(content.split(":")[1:]).strip()
        else:
            display = content
        # D97: her own words, never to be read as a fact about him. D105: every other source says itself too.
        kind, label = source_mark(row, key)
        if kind:
            display = f"{label} {display}"
            marks.add(kind)

        parts = [f"* {display}"]
        if row.get("source_platform"):
            parts.append(f"[{row['source_platform']}]")
        created = row["created_at"]
        parts.append(f"({created.date().isoformat()}, {_age_days(created, now)}d ago)")
        lines.append(" ".join(parts))

        if len(lines) >= CONTEXT_CAP:
            break
    return "\n".join(lines), frozenset(marks)


def _age_days(created_at: datetime, now: datetime) -> int:
    """Age in whole days, JS-style: Math.round((now - created) / 86400000).

    math.floor(x + 0.5) reproduces JS Math.round (half rounds UP) — Python's
    round() banker's-rounds halves to even, which would drift on .5 boundaries.
    """
    return math.floor((now - created_at).total_seconds() / 86400 + 0.5)


def openrouter_embedder(api_key: str, timeout_s: float = 30.0) -> Embedder:
    """The real embed seam — OpenRouter's OpenAI-compatible embeddings endpoint.

    n8n mapping: the "Generate Embedding" HTTP Request node (header auth, same
    model). stdlib urllib keeps this dependency-free; only parity_check and the
    future live wiring call it — offline tests inject a fake Embedder instead.
    """

    def embed(text: str) -> list[float]:
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/embeddings",
            data=json.dumps({"model": EMBED_MODEL, "input": text}).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            payload = json.load(resp)
        return payload["data"][0]["embedding"]

    return embed
