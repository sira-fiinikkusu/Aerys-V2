"""Discord slash-command handlers — the memory/profile chains of n8n 03-02, ported pure.

n8n mapping (03-02 Discord Slash Commands, OGqzkWSpgK9XN8ZM):

  /aerys-status  -> "Get Linked Status" + "Format Status"        -> status()
  /aerys-profile -> "Update Profile Name"                        -> set_profile_name()
  /aerys-recall  -> "Recall Query" + "Format Recall"             -> recall()
  /aerys-forget  -> "Forget Search" -> Override API 'retract'    -> forget()
  /aerys-correct -> "Correct Search" -> Override API 'correct'   -> correct()
  /aerys-tell    -> Override API 'add'                           -> tell()
  /aerys-pin     -> DROPPED. The old lock lived on core_claim.locked; V2's
                    `memories` table has no locked column, so there is nothing
                    to pin. Deliberate — not an oversight.

SEMANTIC PORT, not a SQL port: the old chains acted on V1's `core_claim` table
(status/sensitivity/confidence/locked) through the Override API webhook. V2's
live store is the `memories` table, so each command's INTENT is re-expressed
against it directly:

  - "retract" -> soft-delete (SET deleted_at = now()), the only delete V2 does.
  - "correct" -> the atomic CTE soft-delete + insert from CLAUDE.md ("atomic row
    replacement"), same shape as extraction's TRIAGE_REPLACE_SQL. The new row
    KEEPS the old created_at — created_at means "when this fact first landed"
    (the h.created_at lesson), and a correction does not reset history.
  - "add" -> plain insert with key_label 'user.stated' (the old flow appended a
    Date.now() suffix purely to dodge core_claim's uniqueness; the `memories`
    live-uniqueness index is per (person_id, key_label) and /aerys-tell facts
    are free-form, so V2 keeps the label clean).
  - locked checks -> gone with the column.

Every write also writes audit_log ('user' actor) — the Override API did that
server-side in V1; here it's inline, same transaction scope as the write.

DATABASE: every handler expects the PROD `aerys` connection (tables: memories,
persons, platform_identities, audit_log) — NOT the brain's own aerys_v2 DB
(v2_* tables). The caller injects it; nothing here connects.

PURITY / FAILURE POSTURE: pure functions, conn + embedder injected, no I/O at
import time. DB exceptions RAISE (the Discord binding layer catches and turns
them into an apology — that's its job, not ours). The ONE handled failure is a
raising embedder in correct()/tell(): the embed runs BEFORE any write, so a
dead embedding API means zero rows touched and an honest "couldn't store that
right now" reply — never a memory without a vector.

ILIKE parity note: forget/correct search with '%' + fact + '%' exactly like the
n8n nodes — user-supplied % / _ wildcards pass through unescaped, same as V1.
An EMPTY fact is rejected here though (V1's '%%' matched an arbitrary row —
that's a footgun on a destructive command, not a feature).
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Sequence

from aerys_v2.services.memory import embedding_to_pgvector

Embedder = Callable[[str], Sequence[float]]

# --- SQL ----------------------------------------------------------------------

# n8n "Recall Query", re-aimed at memories: live rows only, newest activity
# first. COALESCE because rows inserted before updated_at existed may be NULL.
RECALL_SQL = """\
SELECT content, privacy_level
FROM memories
WHERE person_id = %(person_id)s::uuid AND deleted_at IS NULL
ORDER BY COALESCE(updated_at, created_at) DESC
LIMIT 10
"""

# n8n "Forget Search" / "Correct Search": first live row whose content matches.
# Selects everything correct() must carry onto the replacement row.
FIND_MEMORY_SQL = """\
SELECT id, content, key_label, context, privacy_level, created_at
FROM memories
WHERE person_id = %(person_id)s::uuid AND deleted_at IS NULL
  AND content ILIKE %(pattern)s
LIMIT 1
"""

# The Override API's 'retract', V2-style: soft delete, never a hard DELETE.
# The AND person_id belt is defense-in-depth: the id always comes from a
# person-scoped find above, but the person boundary should be structural,
# not call-site-dependent (review finding, 2026-07-11).
FORGET_SQL = """\
UPDATE memories SET deleted_at = now()
WHERE id = %(id)s AND person_id = %(person_id)s::uuid
"""

# The Override API's 'correct', V2-style: the atomic CTE replacement (CLAUDE.md
# pattern, same shape as extraction's TRIAGE_REPLACE_SQL). One statement = one
# transaction — no window where the old row is gone and the new one isn't in.
# created_at is the OLD row's (when the fact first landed); updated_at=now()
# marks the correction. event_date is deliberately not carried: the corrected
# value may no longer describe the same dated event, and guessing is worse
# than blank.
CORRECT_REPLACE_SQL = """\
WITH soft_del AS (
  UPDATE memories SET deleted_at = now()
  WHERE id = %(old_id)s AND person_id = %(person_id)s::uuid
  RETURNING id
)
INSERT INTO memories
  (person_id, content, key_label, context, embedding,
   source_platform, privacy_level, created_at, updated_at)
VALUES
  (%(person_id)s::uuid, %(content)s, %(key_label)s, %(context)s,
   %(embedding)s::vector, 'discord', %(privacy_level)s,
   %(created_at)s, now())
"""

# The Override API's 'add', V2-style. created_at = now(): the user is stating
# the fact right now, so "when this fact first landed" IS now. key_label is a
# param: prod's live-uniqueness index is (person_id, key_label) WHERE
# deleted_at IS NULL, so a bare 'user.stated' works exactly once per person
# then UniqueViolations forever (review finding, 2026-07-11) — each tell gets
# a unique 'user.stated.<suffix>' label, same dodge V1's Date.now() suffix did
# on core_claim, just admitted in the docstring instead of buried.
TELL_INSERT_SQL = """\
INSERT INTO memories
  (person_id, content, key_label, embedding,
   source_platform, privacy_level, created_at, updated_at)
VALUES
  (%(person_id)s::uuid, %(content)s, %(key_label)s, %(embedding)s::vector,
   'discord', %(privacy_level)s, now(), now())
"""

# n8n "Get Linked Status", verbatim intent (positional $1 -> named param).
STATUS_SQL = """\
SELECT pi.platform, pi.platform_user_id, pi.username, p.display_name
FROM platform_identities pi
JOIN persons p ON p.id = pi.person_id
WHERE pi.person_id = %(person_id)s::uuid
"""

# n8n "Update Profile Name", verbatim intent.
SET_NAME_SQL = """\
UPDATE persons SET display_name = %(name)s WHERE id = %(person_id)s::uuid
"""

AUDIT_SQL = """\
INSERT INTO audit_log (who, action, details) VALUES ('user', %(action)s, %(details)s::jsonb)
"""


# --- helpers ------------------------------------------------------------------


def _audit(conn: Any, action: str, details: dict) -> None:
    conn.execute(AUDIT_SQL, {"action": action, "details": json.dumps(details)})


def _find_memory(conn: Any, person_id: str, fact: str) -> tuple | None:
    """First live memory whose content ILIKE-matches `fact`, or None.

    Returns the raw row tuple: (id, content, key_label, context,
    privacy_level, created_at).
    """
    return conn.execute(
        FIND_MEMORY_SQL,
        {"person_id": person_id, "pattern": "%" + fact + "%"},
    ).fetchone()


def _not_found(fact: str) -> str:
    # Honest reply NAMING the search — the V1 "I couldn't find a memory matching
    # that." left the user guessing what "that" was.
    return f'I couldn\'t find a memory matching "{fact}".'


_EMBED_DOWN = (
    "I couldn't store that right now — my embedding service isn't answering. "
    "Nothing was changed; try again in a moment."
)


# --- handlers -----------------------------------------------------------------


def recall(conn: Any, person_id: str) -> str:
    """/aerys-recall — n8n "Recall Query" + "Format Recall".

    Up to 10 live memories, most recently touched first. Private rows get a
    lock emoji (the V1 lock marker, repurposed: V2's 'locked' concept is gone,
    privacy is what's worth flagging in the list now).
    """
    rows = conn.execute(RECALL_SQL, {"person_id": person_id}).fetchall()
    if not rows:
        return "I don't have any stored memories about you yet."
    lines = []
    for content, privacy_level in rows:
        lock = " \U0001f512" if privacy_level == "private" else ""
        lines.append(f"• {content}{lock}")
    return "**What I remember about you:**\n" + "\n".join(lines)


def forget(conn: Any, person_id: str, fact: str) -> str:
    """/aerys-forget — n8n "Forget Search" -> Override API 'retract'.

    Soft-deletes the first live match and audits it. The V1 locked-row refusal
    is gone with the locked column.
    """
    if not fact or not fact.strip():
        return "Tell me what to forget — I need something to search for."
    row = _find_memory(conn, person_id, fact)
    if row is None:
        return _not_found(fact)
    memory_id, content = row[0], row[1]
    conn.execute(FORGET_SQL, {"id": memory_id, "person_id": person_id})
    _audit(
        conn,
        "memory.forget",
        {"person_id": person_id, "memory_id": str(memory_id), "content": content},
    )
    return f"Done — I've forgotten: **{content}**"


def correct(
    conn: Any, person_id: str, fact: str, value: str, embedder: Embedder
) -> str:
    """/aerys-correct — n8n "Correct Search" -> Override API 'correct'.

    Finds like forget(), embeds the NEW value first (a raising embedder means
    zero writes), then atomically replaces the row via the soft-delete CTE.
    The replacement keeps key_label/context/privacy_level/created_at from the
    old row — identity and history survive, only the content changes.
    """
    if not fact or not fact.strip():
        return "Tell me which memory to correct — I need something to search for."
    row = _find_memory(conn, person_id, fact)
    if row is None:
        return _not_found(fact)
    old_id, old_content, key_label, context, privacy_level, created_at = row
    try:
        embedding = embedding_to_pgvector(embedder(value))
    except Exception:
        return _EMBED_DOWN
    conn.execute(
        CORRECT_REPLACE_SQL,
        {
            "old_id": old_id,
            "person_id": person_id,
            "content": value,
            "key_label": key_label,
            "context": context,
            "embedding": embedding,
            "privacy_level": privacy_level,
            "created_at": created_at,
        },
    )
    _audit(
        conn,
        "memory.correct",
        {
            "person_id": person_id,
            "memory_id": str(old_id),
            "old_content": old_content,
            "new_content": value,
        },
    )
    return f"Updated — **{old_content}** is now **{value}**."


def tell(
    conn: Any, person_id: str, fact: str, embedder: Embedder, privacy_level: str
) -> str:
    """/aerys-tell — the Override API 'add', V2-style.

    Inserts a fresh 'user.stated.<suffix>' memory (unique suffix per insert —
    the live-uniqueness index means a repeated bare label would refuse the
    second tell forever). privacy_level is the CALLER's call — the room decides
    ('private' from a DM, 'public' from a guild channel), the same
    room-not-person rule as resolver._privacy_for. Embeds before writing:
    a raising embedder means no row and an honest reply.
    """
    if not fact or not fact.strip():
        return "Tell me the thing to remember — I got an empty fact."
    try:
        embedding = embedding_to_pgvector(embedder(fact))
    except Exception:
        return _EMBED_DOWN
    key_label = f"user.stated.{uuid.uuid4().hex[:8]}"
    conn.execute(
        TELL_INSERT_SQL,
        {
            "person_id": person_id,
            "content": fact,
            "key_label": key_label,
            "embedding": embedding,
            "privacy_level": privacy_level,
        },
    )
    _audit(
        conn,
        "memory.tell",
        {"person_id": person_id, "key_label": key_label, "content": fact,
         "privacy_level": privacy_level},
    )
    return f"Got it — I'll remember: **{fact}**"


def status(conn: Any, person_id: str) -> str:
    """/aerys-status — n8n "Get Linked Status" + "Format Status".

    Display name + every linked platform account. The n8n row filter
    (`.filter(r => r.platform)`) existed only to drop the UNION ALL zero-row
    sentinel — Python just checks for an empty result.
    """
    rows = conn.execute(STATUS_SQL, {"person_id": person_id}).fetchall()
    display_name = rows[0][3] if rows and rows[0][3] else "Unknown"
    linked = ", ".join(
        f"{platform} ({username or platform_user_id})"
        for platform, platform_user_id, username, _ in rows
    )
    return f"**Identity:** {display_name}\n**Linked platforms:** {linked or 'none'}"


def set_profile_name(conn: Any, person_id: str, name: str) -> str:
    """/aerys-profile name — n8n "Update Profile Name", plus the confirm reply
    the workflow buried in an HTTP node body."""
    if not name or not name.strip():
        return "Give me an actual name to call you — I got an empty one."
    name = name.strip()
    conn.execute(SET_NAME_SQL, {"name": name, "person_id": person_id})
    _audit(conn, "profile.set_name", {"person_id": person_id, "display_name": name})
    return f"Done — I'll call you **{name}** from now on."
