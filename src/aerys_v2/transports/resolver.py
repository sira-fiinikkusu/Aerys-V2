"""Identity resolution for transports — the AUTH BOUNDARY.

n8n mapping: the create-on-miss / who-is-this half of 03-01 Identity Resolver
(f3eDUPbif0RnhKIn). The READ half already lives in services/identity.resolve_identity
(a SELECT). This module turns that lookup into the transport-facing decision every
inbound message needs: WHO is this platform account, and what may Aerys tell them?

The one invariant that makes this file reviewed-not-delegated:

    A platform account NOT in platform_identities resolves COLD — a non-UUID
    user_id that build_context structurally refuses to hydrate (see context._is_uuid).
    It NEVER inherits the owner's person_id. The codebase has exactly ONE
    owner-passthrough (the Bearer-authed HTTP/voice pipe in http_api.py, a
    deliberately single-user channel) and it lives nowhere near here. A stranger,
    or any second user, cannot become Chris and cannot read Chris's memories.

Second invariant: privacy_context follows the ROOM, not the person. A 1:1 DM is
'private'; a group/guild is 'public'. The profile service's visibility gates use
this to keep even the OWNER's dm-only claims out of a shared room.

Failure posture: a dead database resolves EVERYONE cold (identical to a stranger),
logged, never raised into the transport. Degrading to "I don't know you" is safe;
degrading to "you're the owner" would be catastrophic — so that path does not exist.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from aerys_v2.services.identity import resolve_identity
from aerys_v2.state import Identity

log = logging.getLogger(__name__)


def _privacy_for(channel_kind: str) -> str:
    """Room-scoped privacy: only a true 1:1 DM is private; every shared room is public."""
    return "private" if channel_kind == "dm" else "public"


def identity_from_lookup(row: dict | None, event: Any) -> Identity:
    """PURE: turn a resolve_identity result into the transport Identity.

    `row` is the dict resolve_identity returns when the account is known, or None
    for a stranger (and, by the db_resolver's design, for a DB failure). The entire
    boundary is this one branch, kept pure so the safety invariants are unit-tested
    with zero I/O:

      - known  -> user_id IS the real person_id (a UUID string); the chat/action
                  node hands it to build_context, which hydrates profile + memories.
      - None   -> user_id is a COLD, non-UUID handle ("{platform}:{id}"); build_context
                  sees a non-UUID and returns '' — no profile, no memories, no owner.

    display_name prefers the DB's canonical name and falls back to whatever the
    platform showed. privacy_context always rides along, derived from the room.
    """
    privacy = _privacy_for(event.channel_kind)
    room = getattr(event, "channel_name", "") or ""  # "" for DMs / events without the field
    # The room's WHERE, carried onto identity so it survives person-keyed threading
    # (thread_id is now 'person:{id}' and no longer encodes the surface). getattr with
    # defaults keeps the pure mapper working for minimal test-event fakes that only
    # pin the fields a given test cares about.
    surface = {
        "platform": getattr(event, "platform", "") or "",
        "channel_kind": getattr(event, "channel_kind", "") or "",
        "channel_id": str(getattr(event, "channel_id", "") or ""),
    }
    if row is None:
        return {
            "user_id": f"{event.platform}:{event.platform_user_id}",
            "display_name": event.display_name,
            "privacy_context": privacy,
            "channel_name": room,
            **surface,
        }
    return {
        "user_id": row["person_id"],
        "display_name": row.get("display_name") or event.display_name,
        "privacy_context": privacy,
        "channel_name": room,
        **surface,
    }


def db_resolver(
    memories_database_url: str,
    *,
    connect: Callable[..., Any] | None = None,
) -> Callable[[Any], Identity]:
    """Build the DB-backed resolve_fn the transports inject.

    Opens a READ-ONLY connection per call against the prod `aerys` database — same
    seam and same one-conn-per-turn tradeoff as factory.context_fn_for (personal-
    assistant volume, ~1ms LAN roundtrip to the NAS; a pool is a drop-in swap behind
    this seam). `connect` is injectable so the shell is testable with a fake
    connection; production passes psycopg.connect.

    ANY exception (NAS down, DNS hiccup, malformed row) resolves the caller COLD via
    identity_from_lookup(None, event) — logged, never raised. Per the module
    docstring: the safe degradation is 'stranger', and 'owner' is not reachable
    from this function at all.
    """
    if connect is None:
        import psycopg

        connect = psycopg.connect

    def resolve(event: Any) -> Identity:
        try:
            with connect(memories_database_url) as conn:
                conn.read_only = True
                row = resolve_identity(conn, event.platform, event.platform_user_id)
            return identity_from_lookup(row, event)
        except Exception:
            log.warning(
                "identity resolve failed for %s:%s — resolving cold",
                getattr(event, "platform", "?"),
                getattr(event, "platform_user_id", "?"),
                exc_info=True,
            )
            return identity_from_lookup(None, event)

    return resolve
