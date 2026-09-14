"""Read the room she is standing in, not only the part addressed to her.

`room_context.py` reads `v2_turns`, and a message becomes a turn only when it is
addressed to her. So in a channel she could see her own conversations and nothing
else. On 2026-09-13 Chris sent two messages seconds apart in #resonance; only the
one that mentioned her became a turn, and when he asked what he had said before it
she answered from their last real exchange — accurate about a view of the room that
was missing most of the room.

This reads the channel from Discord itself, at the moment she is summoned. That is
the proportionate shape: she looks at the room when spoken to, the way a person
would, rather than recording everyone's conversation all the time. Nothing new is
stored; `v2_turns` stays the audit spine of what she actually answered.

Degrade-safe like every other context seam: no client, a closed loop, a gateway
hiccup or a slow fetch all fall back to the turns-table reader, or to nothing.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from aerys_v2.services.room_context import PUBLIC_CHANNELS, format_room_messages

log = logging.getLogger(__name__)

#: Messages of history to read. The turns-table reader's limit counts exchanges;
#: this counts single messages, so it is larger for the same amount of room.
DEFAULT_LIMIT = 30

#: A room block is worth a moment, never a stall: she answers late or not at all.
DEFAULT_TIMEOUT_S = 3.0


class LiveRoomReader:
    """A `room_context_fn` backed by the live channel, with the DB one as fallback.

    Built before the gateway exists (the graph needs it at construction), then
    `attach()`ed to the client once it does.
    """

    def __init__(self, fallback: Callable[[str, str], str] | None = None, *,
                 limit: int = DEFAULT_LIMIT, timeout_s: float = DEFAULT_TIMEOUT_S,
                 speaker_names: dict | None = None):
        self._fallback = fallback
        self._limit = limit
        self._timeout_s = timeout_s
        # {platform_user_id: canonical name}. The turns table already carries the
        # resolved name, so a TURN of his says "Chris"; the live room read is the one
        # path that bypasses identity resolution and shows whatever Discord shows.
        # On 2026-09-14 that had her tell him "the word came from Sira, not from you"
        # — Sira being his own Discord display name. Resolved ONCE at startup rather
        # than per message: a room read is thirty messages and must not become thirty
        # queries. Keyed on the ACCOUNT ID, never the display name, because a display
        # name is not an identity and anyone can set theirs to his.
        self._speaker_names = dict(speaker_names or {})
        self._client = None

    def attach(self, client) -> None:
        self._client = client

    def _fall_back(self, channel_id: str, channel: str) -> str:
        if self._fallback is None:
            return ''
        try:
            return self._fallback(channel_id, channel)
        except Exception:
            log.warning('room-context fallback failed for channel %s', channel_id, exc_info=True)
            return ''

    def __call__(self, channel_id: str, channel: str) -> str:
        # Public surfaces only, exactly as the turns-table reader enforces: there is
        # no path for a DM to be read as a room.
        if channel not in PUBLIC_CHANNELS:
            return ''
        client, loop = self._client, getattr(self._client, 'loop', None)
        if client is None or loop is None or getattr(loop, 'is_closed', lambda: True)():
            return self._fall_back(channel_id, channel)
        try:
            future = asyncio.run_coroutine_threadsafe(self._read(client, channel_id), loop)
            rows = future.result(timeout=self._timeout_s)
        except Exception:
            log.warning('live room read failed for channel %s; using the turns table',
                        channel_id, exc_info=True)
            return self._fall_back(channel_id, channel)
        return format_room_messages(rows) or self._fall_back(channel_id, channel)

    async def _read(self, client, channel_id: str):
        channel = client.get_channel(int(channel_id))
        if channel is None:
            channel = await client.fetch_channel(int(channel_id))
        rows = []
        async for message in channel.history(limit=self._limit):
            author = getattr(message, 'author', None)
            name = getattr(author, 'display_name', None) or getattr(author, 'name', None)
            known = self._speaker_names.get(str(getattr(author, 'id', '')))
            rows.append((known or name, getattr(message, 'content', '')))
        rows.reverse()  # history yields newest first; the block reads chronologically
        return rows
