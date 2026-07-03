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
No similarity threshold: ALL of a person's live memories are scored, top 20 kept.

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
RETRIEVE_LIMIT = 20  # rows scored & returned by SQL
CONTEXT_CAP = 5  # lines that survive dedup into the prompt block
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
    ))) AS combined_score
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


def format_memory_context(rows: list[dict], *, now: datetime | None = None) -> str:
    """Build the prompt-ready memory block — the "Format Memory Context" Code node.

    Pipeline preserved exactly:
      1. drop rows with no content (was the sentinel-row filter; kept as safety)
      2. dedup by key: text before the first ':' (stripped, lowered) when a colon
         exists, else the whole content — first occurrence wins, and rows arrive
         score-sorted DESC so the winner is the best-scored duplicate
      3. cap at CONTEXT_CAP (5) lines
      4. per line: display = text after the FIRST ':' (rest rejoined, so values
         containing colons — URLs, timestamps — survive), else full content;
         then '[source_platform]' if present, then '(YYYY-MM-DD, Nd ago)'
    Returns '' for zero rows (caller injects nothing into the prompt).

    Date added vs the n8n Format node (which emitted only '(Nd ago)'): "when
    did X happen?" questions need the calendar date, and the model can't
    recover it from a bare day-count without knowing today's date.

    Cosmetic fix vs n8n: the JS template `* ${display} ${src} ${age}` left an
    interior double space when src was missing; we join parts with single spaces.
    """
    now = now or datetime.now(timezone.utc)
    seen: set[str] = set()
    lines: list[str] = []
    for row in rows:
        content = row.get("content")
        if content is None:
            continue
        key = (
            content.split(":")[0].strip().lower()
            if ":" in content
            else content.strip().lower()
        )
        if key in seen:
            continue
        seen.add(key)

        if ":" in content:
            display = ":".join(content.split(":")[1:]).strip()
        else:
            display = content

        parts = [f"* {display}"]
        if row.get("source_platform"):
            parts.append(f"[{row['source_platform']}]")
        created = row["created_at"]
        parts.append(f"({created.date().isoformat()}, {_age_days(created, now)}d ago)")
        lines.append(" ".join(parts))

        if len(lines) >= CONTEXT_CAP:
            break
    return "\n".join(lines)


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
