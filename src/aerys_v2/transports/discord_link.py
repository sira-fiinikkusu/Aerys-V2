"""Discord identity slash commands — the WRITE half of cross-platform linking.

n8n mapping: the /link, /unlink, /admin-link and /admin-unlink chains of
03-02 Discord Slash Commands (OGqzkWSpgK9XN8ZM). The SQL is ported essentially
verbatim from the workflow's Postgres nodes ("Store Link Code", "Lookup Link
Code", "Merge Platform Identities" ... "Delete Platform Identity"); the reply
texts stay close to the originals.

DATABASE: every handler takes a connection to the PROD ``aerys`` database
(persons, platform_identities, pending_links, conversations, messages,
memories, audit_log) — NOT the brain's own aerys_v2 DB (v2_* tables). Passing
the wrong conn fails loudly on missing tables, never silently.

TRANSACTION CONTRACT: handlers run their statements sequentially on the
injected conn and never call commit()/rollback() themselves. The binding
layer passes an autocommit-off connection, commits on success, and rolls back
on any raised exception — so a merge that dies halfway (e.g. after moving
conversations but before moving memories) leaves the person graph untouched.
This module is PURE apart from the conn: exceptions raise, no Discord I/O,
no logging side effects.

AUTH BOUNDARY invariants (same law as transports/resolver.py):
  - A failed lookup NEVER falls through to any other person's id. An expired
    or unknown code returns a human reply and executes ZERO merge statements.
  - The admin role check is the binding layer's job (it passes ``is_admin``);
    an unauthorized call executes ZERO SQL.
  - Merges only ever move the CALLER's rows into the person who provably
    issued the code (or, for admin, between two explicitly looked-up
    identities). There is no owner fallback anywhere in this module.

n8n quirk-workarounds killed here (see services/identity.py for the pattern):
  - ``COALESCE((SELECT ...), NULL)`` scalar-subquery sentinels — n8n drops
    items on zero rows so the workflow forced one; Python checks ``is None``.
  - ``person_id::text`` casts — existed for n8n's IF-node UUID handling; we
    str() in code instead.
  - The per-call ``DELETE FROM pending_links WHERE expires_at < NOW()`` sweep
    that 03-01 ran inside every read — V2 keeps expiry sweeping OUT of hot
    paths. sweep_expired() below is for a maintenance cron only.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

# n8n used Math.random because crypto was unavailable in its sandbox; Python
# has secrets, so the default generator is actually unguessable. Same alphabet:
# no I/O/0/1 lookalikes.
CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 6
CODE_TTL = timedelta(minutes=10)

# The verbatim admin-DM text from the n8n "Send Discord DM" node.
ADMIN_LINK_DM_TEXT = (
    "An admin has linked your Discord account to a Telegram account. "
    "Your conversation history is now unified."
)

# ---- SQL (ported from the n8n Postgres nodes, named-param style) -----------

# n8n "Store Link Code" — platform is hardcoded 'discord' there too: the code
# is always issued FROM the Discord side and redeemed from the other platform.
STORE_LINK_CODE_SQL = """\
INSERT INTO pending_links (code, person_id, platform, expires_at)
VALUES (%(code)s, %(person_id)s, 'discord', %(expires_at)s)
"""

# n8n "Lookup Link Code" (::text cast dropped — we str() in code).
LOOKUP_LINK_CODE_SQL = """\
SELECT pl.person_id, pl.platform
FROM pending_links pl
WHERE pl.code = %(code)s AND pl.expires_at > NOW()
"""

# n8n "Merge Platform Identities" / "Merge Conversations" / "Merge Messages" /
# "Merge Memories" / "Soft Delete Loser Person" / "Delete Pending Link".
# Winner = the person who ISSUED the code; loser = the caller redeeming it.
MERGE_PLATFORM_IDENTITIES_SQL = (
    "UPDATE platform_identities SET person_id = %(winner)s WHERE person_id = %(loser)s"
)
MERGE_CONVERSATIONS_SQL = (
    "UPDATE conversations SET person_id = %(winner)s WHERE person_id = %(loser)s"
)
MERGE_MESSAGES_SQL = "UPDATE messages SET person_id = %(winner)s WHERE person_id = %(loser)s"
MERGE_MEMORIES_SQL = "UPDATE memories SET person_id = %(winner)s WHERE person_id = %(loser)s"
SOFT_DELETE_LOSER_SQL = "UPDATE persons SET deleted_at = NOW() WHERE id = %(loser)s"
DELETE_PENDING_LINK_SQL = "DELETE FROM pending_links WHERE code = %(code)s"

# n8n "Count Platform Identities" / "Delete Discord Identity". The delete is
# tightened with platform_user_id (the n8n original matched platform+person
# only, which would drop BOTH rows if a person somehow had two accounts on
# one platform — narrowing a DELETE is the safe direction).
COUNT_IDENTITIES_SQL = "SELECT count(*) FROM platform_identities WHERE person_id = %(person_id)s"
DELETE_OWN_IDENTITY_SQL = """\
DELETE FROM platform_identities
WHERE platform = %(platform)s
  AND person_id = %(person_id)s
  AND platform_user_id = %(platform_user_id)s
"""

# n8n "Lookup Both Identities" (COALESCE wrappers dropped — a scalar subquery
# already yields NULL on zero rows; the row itself always exists).
LOOKUP_BOTH_IDENTITIES_SQL = """\
SELECT
  (SELECT pi.person_id FROM platform_identities pi
   WHERE pi.platform = 'discord' AND pi.platform_user_id = %(discord_id)s LIMIT 1)
    AS discord_person_id,
  (SELECT pi.person_id FROM platform_identities pi
   WHERE pi.platform = 'telegram' AND pi.platform_user_id = %(telegram_id)s LIMIT 1)
    AS telegram_person_id
"""

# n8n "Admin Merge PI/Conversations/Messages/Memories" / "Admin Soft Delete
# Loser" / "Admin Delete Pending Links" — ::uuid casts kept verbatim.
# Winner = the Discord person; loser = the Telegram person (n8n "Prepare
# Admin Merge": "Discord person is winner; Telegram is loser").
ADMIN_MERGE_PI_SQL = (
    "UPDATE platform_identities SET person_id = %(winner)s::uuid WHERE person_id = %(loser)s::uuid"
)
ADMIN_MERGE_CONVERSATIONS_SQL = (
    "UPDATE conversations SET person_id = %(winner)s::uuid WHERE person_id = %(loser)s::uuid"
)
ADMIN_MERGE_MESSAGES_SQL = (
    "UPDATE messages SET person_id = %(winner)s::uuid WHERE person_id = %(loser)s::uuid"
)
ADMIN_MERGE_MEMORIES_SQL = (
    "UPDATE memories SET person_id = %(winner)s::uuid WHERE person_id = %(loser)s::uuid"
)
ADMIN_SOFT_DELETE_LOSER_SQL = "UPDATE persons SET deleted_at = NOW() WHERE id = %(loser)s::uuid"
ADMIN_DELETE_PENDING_LINKS_SQL = "DELETE FROM pending_links WHERE person_id = %(loser)s::uuid"

# n8n "Lookup Unlink Target" (COALESCE sentinel dropped) / "Delete Platform Identity".
LOOKUP_UNLINK_TARGET_SQL = """\
SELECT pi.person_id
FROM platform_identities pi
WHERE pi.platform = %(platform)s AND pi.platform_user_id = %(target_id)s
LIMIT 1
"""
ADMIN_DELETE_IDENTITY_SQL = """\
DELETE FROM platform_identities
WHERE platform = %(platform)s AND platform_user_id = %(target_id)s
"""

# The 03-01 sweep, exiled to a cron (see module docstring).
SWEEP_EXPIRED_SQL = "DELETE FROM pending_links WHERE expires_at < NOW()"

AUDIT_SQL = (
    "INSERT INTO audit_log (who, action, details) "
    "VALUES ('user', %(action)s, %(details)s::jsonb)"
)


def _audit(conn: Any, action: str, details: dict) -> None:
    conn.execute(AUDIT_SQL, {"action": action, "details": json.dumps(details)})


def generate_link_code() -> str:
    """Default code_gen: 6 chars from the lookalike-free alphabet (n8n
    "Generate Link Code", upgraded from Math.random to secrets)."""
    return "".join(secrets.choice(CODE_CHARS) for _ in range(CODE_LENGTH))


# ---- /link ------------------------------------------------------------------


def link(
    conn: Any,
    person_id: str,
    platform_user_id: str,
    code: str | None = None,
    code_gen: Callable[[], str] = generate_link_code,
) -> str:
    """/link — issue a verification code, or redeem one and merge identities.

    No code: generate + store a pending link (n8n "Generate Link Code" →
    "Store Link Code") and reply with redemption instructions.

    With code: look it up ("Lookup Link Code"); an expired/unknown code or a
    same-person redemption returns a reply and runs NO merge. Otherwise the
    caller's person is merged INTO the code issuer's person — the full n8n
    chain, in order: platform_identities, conversations, messages, memories,
    soft-delete the loser person, delete the pending link, audit. The caller
    wraps all of it in one transaction (see module docstring).
    """
    if not code:
        new_code = code_gen()
        expires_at = datetime.now(timezone.utc) + CODE_TTL
        conn.execute(
            STORE_LINK_CODE_SQL,
            {"code": new_code, "person_id": person_id, "expires_at": expires_at},
        )
        return (
            f"Your link code is **{new_code}** — it expires in 10 minutes.\n"
            f"From your account on the other platform, send `/link {new_code}` "
            f"to unify your identity across both."
        )

    code = code.upper()
    row = conn.execute(LOOKUP_LINK_CODE_SQL, {"code": code}).fetchone()
    if row is None:
        # AUTH BOUNDARY: unknown/expired code stops HERE — no merge statements.
        return (
            "That code wasn't found or has expired — codes last 10 minutes. "
            "Run `/link` (no code) to generate a fresh one."
        )

    source_person_id = str(row[0])  # str() replaces the n8n ::text cast
    source_platform = row[1]
    if source_person_id == str(person_id):
        return "Those accounts are already linked — you're the same person on both."

    # Winner = code issuer, loser = the redeeming caller (verbatim n8n order).
    params = {"winner": source_person_id, "loser": str(person_id)}
    conn.execute(MERGE_PLATFORM_IDENTITIES_SQL, params)
    conn.execute(MERGE_CONVERSATIONS_SQL, params)
    conn.execute(MERGE_MESSAGES_SQL, params)
    conn.execute(MERGE_MEMORIES_SQL, params)
    conn.execute(SOFT_DELETE_LOSER_SQL, {"loser": str(person_id)})
    conn.execute(DELETE_PENDING_LINK_SQL, {"code": code})
    # Kill EVERY outstanding code of the just-merged loser, not only the one
    # redeemed — a stale code issued in the prior 10 minutes would otherwise
    # stay redeemable and merge a third person INTO the soft-deleted loser
    # (review finding 2026-07-11; the admin chain already did this).
    conn.execute(ADMIN_DELETE_PENDING_LINKS_SQL, {"loser": str(person_id)})
    _audit(
        conn,
        "link_merge",
        {
            "winner_person_id": source_person_id,
            "loser_person_id": str(person_id),
            "source_platform": source_platform,
            "redeemed_by_platform_user_id": platform_user_id,
        },
    )
    return (
        f"Accounts linked! Your Discord and {source_platform} identities are "
        f"now one person — your conversation history is now unified."
    )


# ---- /unlink ----------------------------------------------------------------


def unlink(conn: Any, person_id: str, platform: str, platform_user_id: str) -> str:
    """/unlink — detach this platform account, unless it's the person's only one.

    n8n "Count Platform Identities" → "Check Can Unlink" (count must be > 1)
    → "Delete Discord Identity". A sole identity can't be unlinked: the person
    row and their memories would become unreachable.
    """
    row = conn.execute(COUNT_IDENTITIES_SQL, {"person_id": person_id}).fetchone()
    count = int(row[0]) if row else 0
    if count <= 1:
        return (
            "You can't unlink your only connected platform — your memories "
            "would have no account left to reach them. Link another platform "
            "first with `/link`."
        )
    result = conn.execute(
        DELETE_OWN_IDENTITY_SQL,
        {"platform": platform, "person_id": person_id, "platform_user_id": platform_user_id},
    )
    # The narrowed WHERE (platform_user_id added for safety) opened a new gap
    # the n8n original didn't have: the person-level count can pass while the
    # DELETE matches 0 rows. Claiming success — and writing an audit row — for
    # a mutation that never happened is worse than the honest miss (review
    # finding 2026-07-11). rowcount -1 means the driver can't say; trust the
    # count guard in that case, as before.
    if getattr(result, "rowcount", -1) == 0:
        return (
            f"That {platform} account isn't linked to your identity — "
            f"nothing to unlink."
        )
    _audit(
        conn,
        "unlink",
        {
            "person_id": str(person_id),
            "platform": platform,
            "platform_user_id": platform_user_id,
        },
    )
    return (
        f"Unlinked. This {platform} account is no longer connected to your "
        f"identity — your memories stay with your remaining platform(s)."
    )


# ---- /admin-link ------------------------------------------------------------


def admin_link(conn: Any, discord_id: str, telegram_id: str, is_admin: bool) -> dict:
    """/admin-link — force-merge a Discord and a Telegram identity.

    Returns {"reply": str, "notify": [{"platform", "user_id", "text"}, ...]};
    the binding layer performs the notifications (this module does no I/O
    beyond conn). n8n chain: role gate → "Lookup Both Identities" →
    "Prepare Admin Merge" (Discord person is winner; Telegram is loser) →
    "Check Merge Needed" → the Admin Merge statements → Discord DM.
    """
    if not is_admin:
        # AUTH BOUNDARY: no SQL runs for non-admins.
        return {
            "reply": "You need the Aerys Admin role to use this command.",
            "notify": [],
        }

    row = conn.execute(
        LOOKUP_BOTH_IDENTITIES_SQL,
        {"discord_id": discord_id, "telegram_id": telegram_id},
    ).fetchone()
    discord_person = str(row[0]) if row[0] is not None else None
    telegram_person = str(row[1]) if row[1] is not None else None

    if discord_person is None or telegram_person is None:
        missing = []
        if discord_person is None:
            missing.append(f"Discord user {discord_id}")
        if telegram_person is None:
            missing.append(f"Telegram user {telegram_id}")
        return {
            "reply": (
                f"No identity found for {' or '.join(missing)} — they need to "
                f"have messaged Aerys at least once before linking."
            ),
            "notify": [],
        }

    if discord_person == telegram_person:
        # n8n RespondAdminLinkNoOp branch.
        return {
            "reply": "Those accounts already belong to the same person — nothing to merge.",
            "notify": [],
        }

    # Discord person is winner; Telegram is loser (verbatim n8n order).
    params = {"winner": discord_person, "loser": telegram_person}
    conn.execute(ADMIN_MERGE_PI_SQL, params)
    conn.execute(ADMIN_MERGE_CONVERSATIONS_SQL, params)
    conn.execute(ADMIN_MERGE_MESSAGES_SQL, params)
    conn.execute(ADMIN_MERGE_MEMORIES_SQL, params)
    conn.execute(ADMIN_SOFT_DELETE_LOSER_SQL, {"loser": telegram_person})
    conn.execute(ADMIN_DELETE_PENDING_LINKS_SQL, {"loser": telegram_person})
    _audit(
        conn,
        "admin_link_merge",
        {
            "winner_person_id": discord_person,
            "loser_person_id": telegram_person,
            "discord_id": discord_id,
            "telegram_id": telegram_id,
        },
    )
    return {
        "reply": (
            f"Linked. The Telegram identity was merged into the Discord user's "
            f"person — their conversation history is now unified."
        ),
        "notify": [
            {"platform": "discord", "user_id": discord_id, "text": ADMIN_LINK_DM_TEXT},
        ],
    }


# ---- /admin-unlink ----------------------------------------------------------


def admin_unlink(conn: Any, platform: str, target_id: str, is_admin: bool) -> str:
    """/admin-unlink — detach any platform identity by platform + user id.

    n8n chain: role gate → "Lookup Unlink Target" → "Check Unlink Found" →
    "Delete Platform Identity". Note the n8n original has no only-identity
    guard here (unlike /unlink) — admin is trusted to know; kept verbatim.
    """
    if not is_admin:
        # AUTH BOUNDARY: no SQL runs for non-admins.
        return "You need the Aerys Admin role to use this command."

    row = conn.execute(
        LOOKUP_UNLINK_TARGET_SQL, {"platform": platform, "target_id": target_id}
    ).fetchone()
    if row is None:
        return f"No {platform} identity found for user {target_id} — nothing to unlink."

    person_id = str(row[0])
    conn.execute(ADMIN_DELETE_IDENTITY_SQL, {"platform": platform, "target_id": target_id})
    _audit(
        conn,
        "admin_unlink",
        {"platform": platform, "platform_user_id": target_id, "person_id": person_id},
    )
    return f"Unlinked {platform} account {target_id} from person {person_id}."


# ---- maintenance -------------------------------------------------------------


def sweep_expired(conn: Any) -> int | None:
    """Delete expired pending link codes. Cron-only — NEVER call from a hot
    path (that's the 03-01 mistake this module's docstring describes).
    Returns the deleted-row count when the driver exposes it."""
    cur = conn.execute(SWEEP_EXPIRED_SQL)
    return getattr(cur, "rowcount", None)
