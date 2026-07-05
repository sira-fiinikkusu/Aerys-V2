"""Channel-recent room context — the multi-person half of cross-surface continuity.

Person-keyed checkpointer threads (track/memory-continuity) hold only ONE person's
messages, so in a shared public channel the model would otherwise be blind to what
everyone ELSE just said. This module reads the last N turns of a specific public
channel from v2_turns (ALL people) and formats them into a prompt block, so she
"holds the room" on top of the caller's personal thread.

n8n mapping: the guild adapter's `thread_context` snippet (the last few channel
messages spliced into the Core Agent prompt) — but pulled from the durable v2_turns
audit spine instead of a per-execution Discord fetch, and keyed on channel_id.

Safety: EVERY row this returns happened IN the queried public channel, so it is
public-by-origin — there is no path for a DM/private turn (different channel_id) to
appear here. The chat node injects this block ONLY on public turns; DMs never get it.
Belt-and-braces (cross-review 2026-07-05): the query ALSO whitelists the channel enum
to the genuinely-public surfaces ('guild','telegram_group'), so even if a caller ever
passed a private/DM channel by mistake, no private-origin row could surface. Everything
downstream is degrade-safe: a DB hiccup yields an empty block, never a dead turn
(mirrors services/context.py's graceful contract).
"""

from __future__ import annotations

# The room read. Filters (channel_id, channel) — channel disambiguates a theoretical
# discord-snowflake / telegram-chat-id numeric collision — takes the most recent N,
# and the caller re-orders to chronological for rendering. person_id rides along only
# so the formatter can fall back to a short handle when a row predates display_name.
# The `channel IN (...)` whitelist is a privacy backstop: this block is a PUBLIC-room
# feature, so it must NEVER return a row from a DM/voice/cli/http channel even if the
# channel_id somehow matched — public origin is enforced structurally, not just by the
# caller passing the right channel.
_PUBLIC_CHANNELS = ("guild", "telegram_group")

ROOM_TURNS_SQL = """\
SELECT display_name, person_id, input_text, emitted_reply, created_at
FROM v2_turns
WHERE channel_id = %(channel_id)s AND channel = %(channel)s
  AND channel IN ('guild', 'telegram_group')
  AND input_text IS NOT NULL AND input_text <> ''
ORDER BY created_at DESC
LIMIT %(limit)s
"""

# Cap each rendered field so a wall-of-text message can't blow the room block past a
# sane prompt budget. 300 chars matches CLAUDE.md's thread_context lesson (V1 capped
# snippets at 80 and starved sub-agents of context; 300 is the tuned value).
_FIELD_CAP = 300


def _clip(text: object, cap: int = _FIELD_CAP) -> str:
    s = " ".join(str(text or "").split())  # collapse newlines/runs so one turn = one line
    return s if len(s) <= cap else s[: cap - 1].rstrip() + "…"


def _speaker(display_name: object, person_id: object) -> str:
    """A readable speaker label: the display name if we captured one, else a short
    person handle (last 4 of the UUID) so distinct people stay distinguishable, else
    a neutral 'Someone' for a cold/unresolved turn."""
    name = str(display_name or "").strip()
    if name:
        return name
    pid = str(person_id or "").strip()
    return f"person·{pid[-4:]}" if pid else "Someone"


def format_room_context(rows: list) -> str:
    """Rows (newest-first, as ROOM_TURNS_SQL returns) -> a chronological prompt block.

    Each turn renders as the speaker's line and, when present, her own reply — enough
    for her to follow the room's thread without re-reading every word. Returns '' for
    no rows so the caller injects nothing rather than an empty header.
    """
    if not rows:
        return ""
    lines: list[str] = []
    for display_name, person_id, input_text, emitted_reply, _created_at in reversed(rows):
        text = _clip(input_text)
        if not text:
            continue
        lines.append(f"{_speaker(display_name, person_id)}: {text}")
        reply = _clip(emitted_reply)
        if reply:
            lines.append(f"Aerys: {reply}")
    return "\n".join(lines)
