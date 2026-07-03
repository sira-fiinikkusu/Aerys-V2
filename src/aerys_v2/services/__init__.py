"""Read-only DB services — the n8n READ-path sub-workflows as plain functions.

n8n mapping: three workflows the Core Agent called for every message become three
importable functions, no HTTP hop, no Execute Workflow boundary:

  03-01 Identity Resolver (f3eDUPbif0RnhKIn)  → identity.resolve_identity()
  04-03 Profile API       (kQsn28s7NZFvrlfJ)  → profile.get_profile()
  04-02 Memory Retrieval  (GXaRTmTCTP9XqQxY)  → memory.retrieve_memories()

READ-ONLY BY CONTRACT: every statement in this package is a SELECT. The n8n
originals smuggled writes into the read path (pending_links expiry DELETE,
create-person-on-miss INSERT); those are deliberately NOT ported here — see the
notes in identity.py. A future write service owns them.

Every function takes a psycopg-style connection as its first argument (anything
with `.execute(sql, params)` returning a cursor with `.fetchone()/.fetchall()`).
No ORM, no global connection — tests inject a fake, production injects the real
thing built from Settings.database_url.
"""

from aerys_v2.services.identity import resolve_identity
from aerys_v2.services.memory import (
    format_memory_context,
    retrieve_memories,
)
from aerys_v2.services.profile import get_profile

__all__ = [
    "resolve_identity",
    "get_profile",
    "retrieve_memories",
    "format_memory_context",
]
