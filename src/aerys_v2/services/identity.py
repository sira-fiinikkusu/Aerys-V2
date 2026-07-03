"""Identity lookup — the READ half of n8n 03-01 Identity Resolver (f3eDUPbif0RnhKIn).

READ-ONLY MODULE: SELECT only. The n8n workflow did three things on every call:

  1. `DELETE FROM pending_links WHERE expires_at < NOW()` — a WRITE hidden inside
     the read path, running even for pure lookups. KILLED here: expiry sweeping
     belongs in a maintenance cron, not in the hot path of every message.
  2. The lookup (ported below).
  3. Create-on-miss: INSERT persons + INSERT platform_identities with an
     `ON CONFLICT DO NOTHING` that leaks an orphan `persons` row when two first
     messages race. That's a write — the future write service owns it, and must
     do both inserts in ONE transaction and re-select the winning person_id on
     conflict instead of returning the orphan.

n8n quirk-workarounds killed in the query itself:
  - `UNION ALL SELECT NULL::text, NULL::text ... LIMIT 1` — the zero-row sentinel
    (n8n drops items on 0 rows, so the workflow guaranteed one). It also relied on
    UNION ALL branch order to make LIMIT 1 pick the real row first — fragile.
    Python just checks `row is None`.
  - `person_id::text` cast — existed only because n8n's IF `notEmpty` fails on
    UUID-typed values. We str() in code instead of casting in SQL.
"""

from typing import Any

# Plain lookup — the inner logic of the n8n "Lookup Identity" node, sentinel-free.
IDENTITY_LOOKUP_SQL = """\
SELECT pi.person_id, p.display_name
FROM platform_identities pi
JOIN persons p ON p.id = pi.person_id
WHERE pi.platform = %s AND pi.platform_user_id = %s
"""


def resolve_identity(conn: Any, platform: str, platform_user_id: str) -> dict | None:
    """Look up who a platform account belongs to. Returns None for strangers.

    n8n mapping: the "Lookup Identity" Postgres node + the found/not-found IF
    branch. The n8n contract returned `{person_id, display_name, is_new}`; we
    keep the shape for the found case (is_new is always False here — minting a
    new person is a write, which this read-only service refuses to do). A None
    return is V2's "route to the create-person write path" signal.
    """
    row = conn.execute(IDENTITY_LOOKUP_SQL, (platform, platform_user_id)).fetchone()
    if row is None:
        return None
    return {
        # str() replaces the n8n `::text` cast — psycopg returns uuid.UUID objects
        # and downstream consumers (session keys, JSON) want strings.
        "person_id": str(row[0]),
        "display_name": row[1],
        "is_new": False,
    }
