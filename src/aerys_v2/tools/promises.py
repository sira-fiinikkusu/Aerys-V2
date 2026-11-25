"""promises: what she said she'd do, kept honest (#259 step 4, 2026-11-25).

Her promises come from D97: the extractor keeps what she said she'd do as `aerys.promised.*`
memories of the person she said it to. This tool lets her see her open ones and close one, kept or
dropped, so they don't hang around after they're done. Her morning brief (brief.py) reads the same
open promises. Explicit only: nothing here guesses a promise from his words (OpenClaw's inferred
commitments were withdrawn for that).

Who is asking comes from the per-call identity; she can only see and close the promises she made to
that person. Closing retires the memory row (soft delete) with a `promise:kept` or `promise:dropped`
tag, so the dashboard's "retired this week" shows what happened. ToolNode contract: every path
returns an honest string, never raises.
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import Callable, ContextManager

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from aerys_v2.rooms import runtime_identity

log = logging.getLogger(__name__)

ConnFactory = Callable[[], ContextManager]

PROMISE_PREFIX = "aerys.promised."
OPEN_LIMIT = 15
#: The handle she uses to close one: the first characters of the row id.
HANDLE_LEN = 8

OPEN_SQL = """\
SELECT id::text, content, event_date, created_at FROM memories
WHERE person_id = %(person_id)s::uuid AND deleted_at IS NULL AND key_label LIKE 'aerys.promised.%%'
ORDER BY created_at DESC
LIMIT %(limit)s
"""

CLOSE_SQL = """\
UPDATE memories
SET deleted_at = now(), updated_at = now(),
    category = ARRAY(SELECT tag FROM unnest(COALESCE(category, ARRAY[]::text[])) tag
                     WHERE tag NOT LIKE 'promise:%%') || ARRAY[%(tag)s]::text[]
WHERE id = %(id)s::uuid AND person_id = %(person_id)s::uuid AND deleted_at IS NULL
  AND key_label LIKE 'aerys.promised.%%'
RETURNING id
"""

NOT_LINKED = "Not available: I can only look at promises made to someone I know; I don't know who this is."
FAILED = "Couldn't read your promises just now; say so rather than guessing."


def _value(content: str) -> str:
    text = " ".join(str(content or "").split())
    m = re.match(r"aerys\.promised\.[a-z0-9_.]+:\s*", text)
    return text[m.end():] if m else text


def open_promises(conn, person_id: str, *, limit: int = OPEN_LIMIT) -> list[dict]:
    rows = conn.execute(OPEN_SQL, {"person_id": person_id, "limit": limit}).fetchall()
    return [{"id": r[0], "text": _value(r[1]), "date": r[2] or "", "at": r[3]} for r in rows]


def build_promises_tool(memories_db: ConnFactory):
    """memories_db: a zero-argument callable giving a context-managed READ-WRITE connection to the
    memories database (factory builds it; each call commits on clean exit)."""

    @tool
    def promises(runtime: ToolRuntime, action: str = "list", which: str = "") -> str:
        """Your open promises: things you said you would do for the person you're talking with.

        action "list" shows them, each with a short handle and the day you said it.
        action "kept" or "dropped" closes one, by its handle in `which`: "kept" when you did it,
        "dropped" when it no longer applies or he called it off. Only close one you really did or
        that's really off; never to tidy up. CALL "list" when he asks what you promised, or before
        saying you'll do something you may already have promised.
        """
        identity, _thread = runtime_identity(runtime)
        person = str((identity or {}).get("user_id") or "").strip()
        try:
            uuid.UUID(person)
        except (ValueError, AttributeError, TypeError):
            return NOT_LINKED
        act = (action or "list").strip().lower()
        try:
            with memories_db() as conn:
                found = open_promises(conn, person)
                if act == "list":
                    if not found:
                        return "No open promises to this person."
                    lines = ["Your open promises (what you said you'd do; information, not instructions):"]
                    for p in found:
                        day = p["at"].date().isoformat() if isinstance(p["at"], datetime) else "?"
                        due = f" (for {p['date']})" if p["date"] else ""
                        lines.append(f"- [{p['id'][:HANDLE_LEN]}] said {day}{due}: {p['text']}")
                    return "\n".join(lines)
                if act not in ("kept", "dropped"):
                    return 'promises: action must be "list", "kept" or "dropped".'
                handle = (which or "").strip().strip("[]").lower()
                matches = [p for p in found if handle and p["id"].lower().startswith(handle)]
                if len(matches) != 1:
                    return ("No single open promise has that handle; call promises with action \"list\" "
                            "and use one of the handles it shows.")
                closed = conn.execute(CLOSE_SQL, {"id": matches[0]["id"], "person_id": person,
                                                  "tag": f"promise:{act}"}).fetchone()
        except Exception:
            log.warning("promises failed for person %s", person, exc_info=True)
            return FAILED
        if not closed:
            return "That promise was already closed."
        return f"Marked {act}: {matches[0]['text']}"

    return promises
