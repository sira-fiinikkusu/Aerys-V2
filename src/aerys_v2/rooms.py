"""Rooms — can anyone else read what she answers here?

The owner, 11/14 (D47): his mail and his own phone/watch readings must never land in a public space,
"voice is an exception". A turn's identity already says where it came from (transports/resolver.py:
privacy_context is "private" only for a DM; voice and his /ask lines are private too); a sealed room is
private by membership but has other members.
"""

from __future__ import annotations


def shared_room(identity) -> bool:
    """True when others can read her answer: a guild or group channel, a sealed room, or a turn that
    doesn't say it's private. Voice and his DMs are not shared."""
    ident = identity if isinstance(identity, dict) else {}
    return (ident.get("privacy_context") != "private" or bool(ident.get("sealed_room"))
            or str(ident.get("channel_kind") or "dm") != "dm")


def on_his_face(identity) -> bool:
    """His G2 glasses (the lens surface, voice=False on a glasses: thread): his own screen, so it gets
    the same view as his DMs (D47, 11/14 09:10). The panel and the Sticky stay shared surfaces."""
    from aerys_v2.state import LENS_SURFACE

    ident = identity if isinstance(identity, dict) else {}
    return str(ident.get("surface") or "") == LENS_SURFACE


NOT_HIM, SHARED, UNSURE = "not him", "shared", "unsure"


def his_private_read(identity, thread: str, owner: str | None) -> str | None:
    """May she read one of his own private things (his phone and watch readings) into this turn?
    None = yes: his typed line or a voice she's sure is his, his glasses, or his own wake-up, in a room
    that's only his. Otherwise why not (NOT_HIM, SHARED, UNSURE), so the tool can say so plainly."""
    from aerys_v2.hands_gate import OWNER_ASK, OWNER_TEXT, OWNER_VOICE, asker

    ident = identity if isinstance(identity, dict) else {}
    if not owner or str(ident.get("user_id") or "") != str(owner) or str(thread or "").startswith("kael:"):
        return NOT_HIM
    if shared_room(ident):
        return SHARED
    who = asker(ident, thread, owner)
    own_line = who == OWNER_ASK and (on_his_face(ident) or str(ident.get("platform") or "") == "wake")
    return None if who in (OWNER_TEXT, OWNER_VOICE) or own_line else UNSURE


def runtime_identity(runtime) -> tuple[dict, str]:
    """(identity, thread_id) from a tool's injected runtime; empty when there is none."""
    conf = (getattr(runtime, "config", None) or {}).get("configurable") or {}
    identity = conf.get("identity")
    return (identity if isinstance(identity, dict) else {}), str(conf.get("thread_id") or "")
