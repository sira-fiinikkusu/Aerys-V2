"""search_memory and search_history: her own look past the recalled handful (D95-A, Chris 11/24).

Read-only. Who is asking, where, and what they may see come from the per-call identity
(services/recall_search.scope_for); the model passes only words and dates. ToolNode contract:
every path returns an honest string, never raises.
"""
from __future__ import annotations

import logging
from typing import Callable, ContextManager, Iterable, Sequence

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from aerys_v2.rooms import runtime_identity
from aerys_v2.services.memory import Embedder, search_memories
from aerys_v2.services.recall_search import (
    HISTORY_DEFAULT,
    HISTORY_MAX,
    format_history,
    format_memories,
    memory_allowed,
    memory_people,
    parse_day,
    scope_for,
)
from aerys_v2.services import recall_search
from aerys_v2.services.room_context import people_in_room

log = logging.getLogger(__name__)

#: A connection factory: a zero-argument callable returning a context manager that yields a
#: read-only connection (factory.read_only_db builds the real one).
ConnFactory = Callable[[], ContextManager]

NOT_LINKED = "Not available: I can only search for someone I know; I don't know who this is."
MEMORY_FAILED = "Couldn't search your long-term memory just now; say so rather than guessing."
HISTORY_FAILED = "Couldn't search your past conversations just now; say so rather than guessing."
QUERY_CAP = 200


def build_search_memory_tool(memories_db: ConnFactory, *, turns_db: ConnFactory | None = None,
                             embed: Embedder | None = None, shared_surfaces: Iterable[str] = ()):
    surfaces = frozenset(str(s) for s in shared_surfaces or ())

    @tool
    def search_memory(runtime: ToolRuntime, query: str) -> str:
        """Search your long-term memory (the facts you keep about the person you're talking with)
        by meaning and by words, past the few facts recalled for this message.

        CALL THIS when the facts in front of you don't cover what they ask: "what do you know about
        my…", "did I ever tell you…", "what's the name of…", a person, place, plan, date or
        preference you can't see here. Returns up to 12 dated facts. In a shared room it also covers
        the public facts of the people who have been talking in this room.

        query: the words, name or topic to look for.
        """
        identity, thread = runtime_identity(runtime)
        scope = scope_for(identity, thread, shared_surfaces=surfaces)
        if scope is None:
            return NOT_LINKED
        q = " ".join((query or "").split())[:QUERY_CAP]
        if not q:
            return "search_memory needs a query: a name, a topic or a phrase to look for."
        vector = None
        if embed is not None:
            try:
                vector = embed(q)
            except Exception:
                log.warning("search_memory: embedding failed; word match only", exc_info=True)
        try:
            present: list[str] = []
            if scope.in_room and not scope.own and turns_db is not None:
                with turns_db() as conn:
                    present = people_in_room(conn, scope.room_id, scope.room_channel)
            people = memory_people(scope, present)
            with memories_db() as conn:
                rows = search_memories(conn, people, scope.levels, q, caller=scope.person_id,
                                       query_embedding=vector)
        except Exception:
            log.warning("search_memory failed for person %s", scope.person_id, exc_info=True)
            return MEMORY_FAILED
        return format_memories([r for r in rows if memory_allowed(r, scope, people)], q, scope)

    return search_memory


def _other_peoples_words(runtime) -> None:
    """D99 (advisor 11/25): a room search can bring back other people's lines, guests' included, into a turn
    that holds house tools. D98 keeps her own past chats tap-free; other people's words get the same mark the
    room's recent activity already gets (factory's room block sets room_sink["outside"] when anyone but the
    caller is in it), so they can inform the turn but never authorize a reach-out (outside_guard.sources_for)."""
    conf = (getattr(runtime, "config", None) or {}).get("configurable") or {}
    sink = conf.get("room_sink")
    if isinstance(sink, dict):
        sink["outside"] = True


def build_search_history_tool(turns_db: ConnFactory, *, test_bodies: Sequence[str] = (),
                              sealed_ids: Iterable[str] = (), shared_surfaces: Iterable[str] = ()):
    surfaces = frozenset(str(s) for s in shared_surfaces or ())
    sealed = frozenset(str(s) for s in sealed_ids or ())
    bodies = list(test_bodies or ())

    @tool
    def search_history(runtime: ToolRuntime, query: str = "", since: str = "", until: str = "",
                       limit: int = HISTORY_DEFAULT) -> str:
        """Search your past conversations: what they said and what you replied, word for word,
        with the date and where it was said.

        CALL THIS when they ask what you told them, what they said, what was said on a date or in
        a room, or about anything older than the recent conversation in front of you: "what did
        you tell me about the router last week?", "what did I say on Tuesday?", "when did we talk
        about the trip?". Quote and date from what it returns; never guess a date or a quote.

        query: words to find (may be empty when since/until are given).
        since, until: optional days, YYYY-MM-DD, Eastern time, both inclusive.
        limit: how many exchanges to return, 1 to 15 (default 8).
        """
        identity, thread = runtime_identity(runtime)
        scope = scope_for(identity, thread, shared_surfaces=surfaces)
        if scope is None:
            return NOT_LINKED
        q = " ".join((query or "").split())[:QUERY_CAP]
        try:
            start, end = parse_day(since), parse_day(until, end=True)
            if start and end and end <= start:
                start, end = parse_day(until), parse_day(since, end=True)
        except (ValueError, OverflowError):   # Codex: "9999-12-31" + a day overflowed
            return "Couldn't search: give since and until as dates like 2026-11-03."
        if not q and not (start or end):
            return "search_history needs a query, or a since/until date range."
        try:
            n = max(1, min(int(limit or HISTORY_DEFAULT), HISTORY_MAX))
        except (TypeError, ValueError):
            n = HISTORY_DEFAULT
        try:
            with turns_db() as conn:
                rows = recall_search.search_history(conn, scope, q, test_bodies=bodies,
                                                    sealed_ids=sealed, since=start, until=end, limit=n)
        except Exception:
            log.warning("search_history failed for person %s", scope.person_id, exc_info=True)
            return HISTORY_FAILED
        if any(str(r.get("person_id") or "") != scope.person_id for r in rows):
            _other_peoples_words(runtime)
        return format_history(rows, q, scope)

    return search_history
