"""Profile context — n8n 04-03 Profile API (kQsn28s7NZFvrlfJ) as a plain function.

READ-ONLY MODULE: SELECT only.

n8n mapping: this was an HTTP webhook (`POST /webhook/aerys-profile-api`) because
the Core Agent could only reach it over the wire. In Python it's a function call —
the webhook plumbing (body parsing, `person_id` falling back to `speaker_id`,
respondToWebhook typeVersion pinning) all evaporates.

n8n quirk-workarounds killed:
  - `WITH params AS (SELECT $1::uuid AS pid, ...)` CTE wrapper — existed only
    because n8n's Postgres node throws "Syntax error near UNION" when $1/$2 is
    referenced twice across a UNION ALL. psycopg named params repeat freely.
  - trailing `UNION ALL SELECT NULL,NULL,...` zero-row sentinel — n8n drops items
    on 0 rows; Python just handles the empty list (→ cold start).

Semantics preserved verbatim from the live query:
  - P0/P1 sensitivity NEVER surfaces — only P2/P3 leave the database.
  - visibility gate: 'all' always; 'server' only in public context; 'dm' only in
    private context.
  - locked claims outrank everything, then confidence DESC with NULLs last.
  - hard cap 15 claims.
"""

from typing import Any

PROFILE_LIMIT = 15

# Categories emit in this fixed order; anything else follows in first-seen order.
CATEGORY_ORDER = ("basic", "interests", "relationship", "emotional", "other")

# Either key_label marks the claim that carries the person's preferred name.
NAME_KEYS = ("basic.name", "name")

# The inner logic of the n8n query, minus the params-CTE and the NULL sentinel.
PROFILE_SQL = f"""\
SELECT cc.core_id, cc.key_label, cc.claim_text, cc.status, cc.locked,
       cc.confidence, cc.sensitivity, cc.visibility
FROM core_claim cc
WHERE cc.speaker_id = %(pid)s::uuid
  AND cc.status IN ('approved', 'provisional')
  AND cc.sensitivity IN ('P2', 'P3')
  AND (cc.visibility = 'all'
    OR (cc.visibility = 'server' AND %(pctx)s = 'public')
    OR (cc.visibility = 'dm' AND %(pctx)s = 'private'))
ORDER BY CASE WHEN cc.locked THEN 0 ELSE 1 END,
         cc.confidence DESC NULLS LAST
LIMIT {PROFILE_LIMIT}
"""

# Column order of PROFILE_SQL — rows come back as tuples, this names them.
_COLUMNS = (
    "core_id",
    "key_label",
    "claim_text",
    "status",
    "locked",
    "confidence",
    "sensitivity",
    "visibility",
)


def get_profile(conn: Any, person_id: str, privacy_context: str = "public") -> dict:
    """Fetch and format a person's profile context for prompt injection.

    Returns the same shape the webhook returned:
    `{"profile": {"display_name": str|None, "lines": [...], "cold_start": bool}}`.
    """
    rows = conn.execute(
        PROFILE_SQL, {"pid": person_id, "pctx": privacy_context}
    ).fetchall()
    claims = [dict(zip(_COLUMNS, row)) for row in rows]
    return format_profile_context(claims)


def format_profile_context(claims: list[dict]) -> dict:
    """Assemble claim rows into the profile block — the "Format Profile Context" Code node.

    Rules preserved exactly:
      - category = key_label before the first '.'; categories emit in CATEGORY_ORDER,
        then any stragglers in first-seen (dict-insertion) order. Within a category,
        claims keep query order (locked first, then confidence DESC).
      - each line is '• ' + claim_text (U+2022 bullet + space).
      - display_name comes from the name claim's text AFTER THE LAST ': ' — the
        two-char separator, not a bare colon, so "Preferred name: Chris" → "Chris"
        and a value containing lone colons (URLs, times) survives intact.
      - zero claims → cold start: no name, no lines, cold_start=True.
    """
    if not claims:
        return {"profile": {"display_name": None, "lines": [], "cold_start": True}}

    display_name: str | None = None
    by_category: dict[str, list[str]] = {}
    for claim in claims:
        key_label = claim["key_label"] or ""
        text = claim["claim_text"] or ""
        if display_name is None and key_label in NAME_KEYS:
            # JS: claim_text.split(': ').pop() — last segment after ': '.
            # (No ': ' present → split returns the whole text, same as JS.)
            display_name = text.split(": ")[-1]
        by_category.setdefault(key_label.split(".")[0], []).append(text)

    # Fixed-order categories first, then whatever else appeared, in arrival order.
    ordered = [c for c in CATEGORY_ORDER if c in by_category]
    ordered += [c for c in by_category if c not in CATEGORY_ORDER]

    lines = [f"• {text}" for cat in ordered for text in by_category[cat]]
    return {
        "profile": {
            "display_name": display_name,
            "lines": lines,
            "cold_start": False,
        }
    }
