"""What she said on her portable bodies, read back into the house body.

The return leg of "1 identity, 1 memory" (Chris, 2026-09-13): *"if i talk to her in
discord, the conversation from portable isnt added to her turns or active context ...
it still feels like 2 separate individuals."*

WHY THIS IS NEEDED AT ALL. Her house conversation is keyed to the PERSON —
`person:<uuid>` — which is exactly why his counting test already carries across Discord
DM, public Discord, Telegram and voice: one thread, many doors into it. Each portable
body instead runs its own thread, `portable:<body_id>`, on the stick, in its own
checkpointer, and its turns arrive here through the door as ordinary `v2_turns` rows
with a `channel_id` of `portable:<body>:<seq>`. They have always been here. Nothing has
ever read them back.

The stick already reads the other direction (aerys-portable `c380e15` for the house,
`2275471` for her other bodies). This is the leg that closes the loop.

WHERE IT IS ALLOWED, and why it is the mirror image of the room block next door. The
room block carries what OTHER people said in a shared channel, so it is fenced to
PUBLIC surfaces. This carries what HE said to her alone on another of her bodies, so it
is fenced to PRIVATE ones: a DM, Telegram, voice, the glasses. Not a room. The stick is
where he talks to her by himself, which makes its history private-origin, and nothing
private-origin reaches a public room in this codebase without the content-privacy judge
clearing it first. Adversarial review caught the first cut injecting this everywhere.

The consequence, stated so nobody rediscovers it as a bug: a portable turn does not
reach a public channel at all. His counting test still passes going stick -> DM ->
public, because her DM reply lands in the person-keyed thread the public turn reads.
Stick straight to a public channel is the case that loses. Lifting it means running the
privacy judge over portable rows at admission time, not widening this prompt.

A10 IS UNTOUCHED. Only `input_text` and `emitted_reply` are read: both already crossed
the door under its admission rules, and no checkpoint or tool output exists here to
read. The door's wire schema is what guarantees that, and the query below never asks
for anything else even if a future schema offered it.
"""

from __future__ import annotations

# Newest-first, re-ordered by the formatter, the same shape as ROOM_TURNS_SQL next
# door. `channel_id LIKE 'portable:%%'` is the only structural marker a portable turn
# carries — the extractor treats channel_id as opaque text — so the person filter is
# what actually enforces the boundary, not the prefix. Both, belt and braces: a future
# second person's stick can never appear in his block.
PORTABLE_TURNS_SQL = """\
SELECT input_text, emitted_reply, channel_id, created_at
FROM v2_turns
WHERE person_id = %(person_id)s
  AND channel_id LIKE 'portable:%%'
  AND input_text IS NOT NULL AND input_text <> ''
ORDER BY created_at DESC
LIMIT %(limit)s
"""

# Same cap as the room block: one turn renders as one line, and a wall of text cannot
# crowd out the rest of the window.
_FIELD_CAP = 300

#: What a body name may contain. The name is interpolated into her prompt inside
#: '[<body>] Chris: ...', and a channel_id is opaque text to everything upstream, so a
#: name carrying ']' or a newline could close the bracket and forge a line of its own.
#: Mirrors the same list on the door and the stick (aerys-portable, 2026-09-13).
NAME_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.'
NAME_MAX = 32

HEADING = (
    "[Recently between Chris and you on one of your PORTABLE bodies — the stick, or "
    "another machine you run from. This did not happen on this body, so never report "
    "it as something you did here; it is the same conversation, continued elsewhere.]"
)


def _clip(text: object, cap: int = _FIELD_CAP) -> str:
    s = " ".join(str(text or "").split())   # collapse newlines so one turn = one line
    return s if len(s) <= cap else s[: cap - 1].rstrip() + "…"


def body_of(channel_id: object) -> str:
    """The body name out of a 'portable:<body>:<seq>' channel id.

    Degrades to 'a portable body' rather than to anything that could read as this one:
    not knowing which body said something is no reason to imply it was this one.
    """
    parts = str(channel_id or "").split(":")
    name = parts[1] if len(parts) > 1 else ""
    clean = "".join(c for c in name if c in NAME_CHARS)[:NAME_MAX]
    return clean or "a portable body"


def format_portable_context(rows) -> str:
    """Rows (newest-first, as PORTABLE_TURNS_SQL returns) -> a chronological block.

    Returns '' for no rows so the caller injects nothing rather than an empty heading.
    Each line names the body, so a line from the stick can never be read as work the
    house body did.
    """
    if not rows:
        return ""
    lines: list[str] = []
    for input_text, emitted_reply, channel_id, _created_at in reversed(list(rows)):
        said = _clip(input_text)
        if not said:
            continue
        where = body_of(channel_id)
        lines.append(f"[{where}] Chris: {said}")
        reply = _clip(emitted_reply)
        if reply:
            lines.append(f"[{where}] Aerys: {reply}")
    if not lines:
        return ""
    return "\n".join(lines)
