"""sticky_note — her handwriting on the household e-ink displays.

Owner-commissioned the night the first reTerminal Sticky came alive
(2026-08-07): the device rendered a clock and sensors and the owner asked,
fairly, whether it was an Aerys surface or just an expensive clock. This tool
is the difference: SHE writes the note that the e-ink shows. The words are
hers to choose — a morning-brief line, a reminder, a good-luck note.

Mechanism: writes a Home Assistant input_text helper; every Sticky that
renders a "FROM AERYS" section subscribes to that helper and refreshes when
it changes. One slot, newest note wins — it's a sticky note, not a feed.

The e-ink surfaces are HOUSEHOLD surfaces (fridge-visible, guest-visible):
the docstring steers her toward content that belongs on a family fridge.
Failure posture: ToolNode contract — honest strings, never raises.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

NOTE_MAX = 255  # input_text hard limit — HA rejects longer values


def build_sticky_note_tool(
    *,
    base_url: str,
    token: str,
    entity_id: str,
    client: httpx.Client | None = None,
):
    """Close over config and return the sticky_note tool (test seam: inject
    an httpx.Client on a MockTransport)."""
    from langchain_core.tools import tool

    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    http = client or httpx.Client(timeout=10.0)

    @tool
    def sticky_note(message: str) -> str:
        """Write a short note to the household e-ink sticky displays.

        CALL THIS TOOL when the owner asks you to put something on the
        sticky / the fridge display / the e-ink note ("leave a note on the
        sticky", "put that on the fridge"), or when YOU want to leave the
        household a brief note worth glancing at.

        message: the note text, under 255 characters. It appears on a
        family-visible surface in the house, so keep it fridge-appropriate:
        short, warm, useful. The words are yours to choose.

        The newest note replaces the old one — one note at a time.
        """
        text = (message or "").strip()
        if not text:
            return "sticky_note needs some text — there is nothing to write."
        truncated = False
        if len(text) > NOTE_MAX:
            text = text[: NOTE_MAX - 1].rstrip() + "…"
            truncated = True
        try:
            r = http.post(
                f"{base}/api/services/input_text/set_value",
                headers=headers,
                json={"entity_id": entity_id, "value": text},
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            return f"The note did NOT reach the display — Home Assistant said: {e}."
        note = " (it was over the 255-character limit, so I trimmed the end)" if truncated else ""
        return f"Note is up on the sticky displays{note}: {text}"

    return sticky_note
