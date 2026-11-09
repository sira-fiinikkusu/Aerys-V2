"""The reach-out desk: every message she starts herself goes through here (board #176, Stage 2).

The owner's rulings, 2026-11-08:
  D4 "3 a day plus briefs as a starting point but this can be adaptive, i'd like for an
     option for me to say chill out or annoy me more (not exactly in those words)."
  D6 "where-ever she detects my pressence (via our new home pressence + bluetooth
     awareness of my phone)"
  D7 "can we not use rigid emoji reactions for this? ... this should be natural language"
  D15 + the meeting rule: nothing spoken over a work meeting; it waits until 10 minutes
     after; quiet hours 21:00-09:00 make it quiet text.

So one desk, with its checks in code, in this order (decide() below):
  1. no repeats (a dedupe key already sent in the last day is dropped);
  2. safety warnings always go;
  3. topics he muted are dropped (an answer he asked for never is);
  4. the night: an answer he asked for, or anything timely, goes as quiet text; the rest
     waits for the morning;
  5. his meetings: timely things go as quiet text; an answer he asked for goes as quiet
     text, or waits for the window if she would have SAID it; the rest waits;
  6. the daily budget (his, in v2_reach_prefs): over it, routine things are dropped and
     timely ones still go.
Then delivery, where he is: the room the house places him in (his phone's Bluetooth plus
the room sensors: the settled room), if that room has a speaker; otherwise his last
surface (a voice device he spoke on minutes ago, Telegram, or a Discord DM). The headline
is what may be spoken; detail only ever goes as text. A spoken message he doesn't answer
within a few minutes gets a text copy. Never over a satellite that's mid-conversation.

The record (v2_reachouts, migration 016) holds every decision with its reason, so "why
did she go quiet?" has an answer. Everything she queues is delivered by the watcher when
its window opens; an answer he asked for is never dropped, only delayed.

A brief (migration 018) is the one thing the desk treats differently: her morning brief
(brief.py) only exists once he's up and in the room it's for, so the night doesn't hold it and
it isn't counted against the budget (D4 "plus briefs"). It's pinned to that room: it waits for
him there and is never spoken anywhere else. It carries the routine items held overnight: they're
reserved for it before she composes (so they can't also go out on their own meanwhile) and
marked 'folded' in the same write that records the brief as sent; if it never goes, they go on
their own at its cutoff. With a brief armed, the night holds them until that cutoff, not 9.
Its text part (the carried items in full) is retried until it lands, like any detail.

Two more knobs (migration 019): a text_only message is never spoken, wherever he is (email:
today he gets a DM per mail, and the desk adds its rules without moving mail into the
speakers); and a safety warning inside quiet hours goes as a text that DOES buzz rather than
out loud, unless the owner turns safety_voice_at_night on (it breaks through either way; by
day it's spoken where he is).

What she says is the private part: never logged (logs carry ids, sources and statuses).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta, timezone
from typing import Callable, Iterable, Protocol
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
URGENCIES = ("routine", "timely", "safety")
BUDGET_MIN, BUDGET_MAX = 0, 8
CHECKINS_MIN, CHECKINS_MAX = 0, 4      # unasked check-ins a day he's said he wants (followups.py)
#: A topic he can mute that covers others: "stop checking up on me" mutes every unasked check-in.
MUTE_COVERS = {"follow-ups": ("appointments", "to-dos")}
DEDUPE_WINDOW = timedelta(hours=24)
BUSY_RETRY = timedelta(minutes=1)
FAIL_BACKOFF = (timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=5),
                timedelta(minutes=10), timedelta(minutes=30))
GIVE_UP_AFTER = 8                     # failed attempts before a NON-owed item is given up
FOLLOW_THROUGH_FOR = timedelta(hours=1)
NAP_RECHECK = timedelta(minutes=2)     # a nap-held message is looked at again this often
NAP_GRACE = timedelta(seconds=10)      # clock skew only: replies are judged by when his turn STARTED
NAP_DEFAULT = timedelta(hours=1)
NAP_MAX = timedelta(hours=3)
COMPANY_DEFAULT = timedelta(hours=6)  # "people are over" ends on its own if he forgets to say they left
COMPANY_MAX = timedelta(hours=24)
ROOM_WAIT = timedelta(minutes=1)      # a message pinned to a room he isn't in is looked at again this often
DETAIL_FOR = timedelta(days=3)        # a spoken message whose detail never went keeps trying this long
TICK_S = 30
MISSED_PREFIX = "In case you missed what I said:"


@dataclass
class ReachEvent:
    source: str                       # who raised it: kael_note, ticket, ticket_notice, ...
    topic: str                        # what it's about: kael, weather, email, ... (mutable by him)
    headline: str                     # what may be spoken aloud
    detail: str = ""                  # text only, never spoken
    urgency: str = "routine"          # routine | timely | safety
    owed: bool = False                # an answer he asked for: never dropped, never budgeted
    dedupe_key: str | None = None
    expires_at: datetime | None = None
    room: str | None = None           # spoken only in this room; it waits for him there
    brief: bool = False               # a brief: outside the budget; the night doesn't hold it
    folds: tuple = ()                 # ids of held items it carries (marked 'folded' when it goes)
    text_only: bool = False           # never spoken, wherever he is (email)
    text_to: str | None = None        # a text_only message's surface (discord | telegram); None = the desk's default


@dataclass
class Reach:
    """One row of the record."""

    id: int
    source: str
    topic: str
    urgency: str
    owed: bool
    headline: str
    detail: str = ""
    dedupe_key: str | None = None
    status: str = "queued"
    not_before: datetime | None = None
    expires_at: datetime | None = None
    reason: str = ""
    surface: str | None = None
    silent: bool = False
    spoken_at: datetime | None = None
    dm_at: datetime | None = None
    delivered_at: datetime | None = None
    acked_at: datetime | None = None
    attempts: int = 0
    created_at: datetime | None = None
    room: str | None = None
    brief: bool = False
    folds: tuple = ()
    text_only: bool = False
    text_to: str | None = None

    def event(self) -> ReachEvent:
        return ReachEvent(source=self.source, topic=self.topic, headline=self.headline, detail=self.detail,
                          urgency=self.urgency, owed=self.owed, dedupe_key=self.dedupe_key,
                          expires_at=self.expires_at, room=self.room, brief=self.brief, folds=tuple(self.folds),
                          text_only=self.text_only, text_to=self.text_to)


@dataclass
class Prefs:
    budget: int = 3
    muted: list = field(default_factory=list)
    hold_since: datetime | None = None   # a nap: he said so at hold_since...
    hold_until: datetime | None = None   # ...and it lasts until he talks to her again, or this
    checkins: int = 2                    # unasked check-ins a day (migration 020)
    company_until: datetime | None = None  # people are over: nothing said aloud until then (020)


@dataclass
class Decision:
    action: str                       # send | queue | drop
    reason: str = ""
    not_before: datetime | None = None
    silent: bool = False
    voice: str = "ok"                 # ok | reroute (text instead) | notify (a quiet HA notification) | wait
    breaks_through: bool = False      # safety: may speak at night or over a meeting (D4)


def muted_topics(prefs: "Prefs") -> set:
    """What he's muted, with each umbrella topic opened up (MUTE_COVERS)."""
    muted = {str(m).strip().lower() for m in prefs.muted}
    for umbrella, covers in MUTE_COVERS.items():
        if umbrella in muted:
            muted.update(covers)
    return muted


def _utc(dt: datetime | None) -> datetime | None:
    return dt.astimezone(timezone.utc) if dt is not None else None


def next_local(now: datetime, hour: int) -> datetime:
    """The next time it is `hour`:00 Eastern after `now` (DST-safe: built in local time)."""
    local = now.astimezone(EASTERN)
    target = datetime.combine(local.date(), time(hour), tzinfo=EASTERN)
    if target <= local:
        target = datetime.combine(local.date() + timedelta(days=1), time(hour), tzinfo=EASTERN)
    return target.astimezone(timezone.utc)


def local_day_start(now: datetime) -> datetime:
    local = now.astimezone(EASTERN)
    return datetime.combine(local.date(), time(0), tzinfo=EASTERN).astimezone(timezone.utc)


def decide(event: ReachEvent, now: datetime, *, prefs: Prefs, sent_today: int, duplicate: bool,
           quiet: bool, meeting_until: datetime | None, morning_hour: int, hold: str = "meeting",
           safety_voice_at_night: bool = False, company: bool = False) -> Decision:
    d = _decide_checks(event, now, prefs=prefs, sent_today=sent_today, duplicate=duplicate, quiet=quiet,
                       meeting_until=meeting_until, morning_hour=morning_hour, hold=hold,
                       safety_voice_at_night=safety_voice_at_night)
    # People are over (he said so): whatever would be said aloud goes as text instead. Safety
    # still speaks; the record says why the rest didn't.
    if company and d.action == "send" and d.voice != "reroute" and not d.breaks_through:
        # A notification on his phone, watch and glasses (the owner: "i can make it so my G2
        # see's the HA notifies"), or text when that leg isn't there.
        return Decision("send", (d.reason + "; " if d.reason else "") + "company: as a notification",
                        silent=d.silent, voice="notify")
    return d


def _decide_checks(event: ReachEvent, now: datetime, *, prefs: Prefs, sent_today: int, duplicate: bool,
                   quiet: bool, meeting_until: datetime | None, morning_hour: int, hold: str,
                   safety_voice_at_night: bool) -> Decision:
    """The desk's checks, in order (module docstring). Pure: every input is given.
    meeting_until is any hold that works like a meeting; `hold` names it in the record
    ("meeting" or "nap"). Safety breaks through every hold, a nap included: the owner's D4
    lets storm warnings break through the night, and a nap is lighter than the night. At
    night it breaks through as a text that buzzes, not out loud in the bedroom, unless
    safety_voice_at_night says to speak it."""
    if duplicate:
        return Decision("drop", "already sent")
    if event.expires_at is not None and not event.owed and now >= _utc(event.expires_at):
        return Decision("drop", "expired before its window")
    if event.urgency == "safety":
        if quiet and not safety_voice_at_night:
            return Decision("send", "safety at night: as text that buzzes", voice="reroute", breaks_through=True)
        return Decision("send", "safety", breaks_through=True)
    if not event.owed and event.topic.strip().lower() in muted_topics(prefs):
        return Decision("drop", f"you muted {event.topic}")
    if quiet and not event.brief:      # a brief exists only once he's up and in its room
        if event.owed or event.urgency == "timely":
            return Decision("send", "night: as quiet text", silent=True, voice="reroute")
        return Decision("queue", "night: waits for the morning", not_before=next_local(now, morning_hour))
    if meeting_until is not None:
        # A meeting ends when the calendar says; a nap can end early (he talks to her), so
        # what it holds is looked at again every couple of minutes instead of at its end.
        if hold == "nap":
            meeting_until = min(meeting_until, now + NAP_RECHECK)
        if event.urgency == "timely":
            return Decision("send", f"{hold}: as quiet text", silent=True, voice="reroute")
        if event.owed:
            return Decision("send", f"{hold}: quiet text, or the next window if it would be spoken",
                            silent=True, voice="wait", not_before=meeting_until)
        after = "until 10 minutes after" if hold == "meeting" else "until he's up"
        return Decision("queue", f"{hold}: waits {after}", not_before=meeting_until)
    if not event.owed and not event.brief and sent_today >= prefs.budget:
        if event.urgency == "timely":
            return Decision("send", f"over today's budget ({prefs.budget}), but timely")
        return Decision("drop", f"over today's budget ({prefs.budget})")
    return Decision("send")


# --- the record --------------------------------------------------------------------------


class ReachStore(Protocol):
    def add(self, event: ReachEvent, now: datetime, *, status: str, reason: str,
            not_before: datetime | None) -> Reach: ...
    def recently_sent(self, dedupe_key: str, since: datetime) -> bool: ...
    def decided(self, dedupe_key: str, since: datetime) -> bool: ...
    def withdraw(self, dedupe_key: str, now: datetime, reason: str = ...) -> bool: ...
    def statuses(self, dedupe_key: str) -> set: ...
    def carried(self, dedupe_key: str, since: datetime) -> str | None: ...
    def add_new(self, event: ReachEvent, now: datetime, *, status: str, reason: str,
                not_before: datetime | None) -> Reach | None: ...
    def foldable(self, since: datetime, until: datetime) -> list[Reach]: ...
    def sent_quietly(self, since: datetime) -> list[Reach]: ...
    def sent_since(self, since: datetime) -> int: ...
    def due(self, now: datetime) -> list[Reach]: ...
    def awaiting_reply(self, before: datetime, after: datetime, detail_after: datetime | None = None) -> list[Reach]: ...
    def mark(self, reach_id: int, now: datetime, **fields) -> None: ...
    def settle(self, reach_id: int, now: datetime, *, folds: tuple = (), **fields) -> None: ...
    def prefs(self, default_budget: int) -> Prefs: ...
    def set_prefs(self, prefs: Prefs, now: datetime, change: str, words: str) -> None: ...


MARKABLE = {"status", "not_before", "reason", "surface", "silent", "spoken_at", "dm_at",
            "delivered_at", "acked_at", "attempts"}


class MemoryReachStore:
    """In-process store (tests, dev boxes). Same semantics as the Postgres one."""

    def __init__(self, rows: Iterable[Reach] = (), prefs: Prefs | None = None):
        self.rows = {r.id: r for r in rows}
        self._next = max(self.rows, default=0) + 1
        self._prefs = prefs
        self.changes: list = []

    def add(self, event, now, *, status, reason, not_before):
        r = Reach(id=self._next, source=event.source, topic=event.topic, urgency=event.urgency, owed=event.owed,
                  headline=event.headline, detail=event.detail, dedupe_key=event.dedupe_key, status=status,
                  not_before=not_before, expires_at=event.expires_at, reason=reason, created_at=now,
                  room=event.room, brief=event.brief, folds=tuple(event.folds), text_only=event.text_only,
                  text_to=event.text_to)
        self.rows[r.id] = r
        self._next += 1
        return r

    def recently_sent(self, dedupe_key, since):
        return any(r.dedupe_key == dedupe_key and r.status in ("sent", "queued") and r.created_at >= since
                   for r in self.rows.values())

    def decided(self, dedupe_key, since):
        """Any record under this key since `since`, whatever became of it (a quiet evening too)."""
        return any(r.dedupe_key == dedupe_key and r.created_at >= since for r in self.rows.values())

    def withdraw(self, dedupe_key, now, reason="he called it off"):
        """Drop what's still queued under this key (he called it off, or something else carries
        it now). True if anything was."""
        hit = [r for r in self.rows.values() if r.dedupe_key == dedupe_key and r.status == "queued"]
        for r in hit:
            r.status, r.reason = "dropped", reason
        return bool(hit)

    def statuses(self, dedupe_key):
        """What became of every record under this key (empty if there's none)."""
        return {r.status for r in self.rows.values() if r.dedupe_key == dedupe_key}

    def carried(self, dedupe_key, since):
        """A 'folded' marker under this key (since `since`) whose carrier, the row that holds it in
        its folds, is still alive: 'sent', 'pending' (queued), or None (no marker, or its carrier
        was dropped, failed or never written)."""
        marks = {r.id for r in self.rows.values() if r.dedupe_key == dedupe_key and r.status == "folded"
                 and r.created_at and r.created_at >= since}
        states = {w.status for w in self.rows.values() if marks & set(w.folds) and w.status in ("queued", "sent")}
        return "sent" if "sent" in states else "pending" if states else None

    def add_new(self, event, now, *, status, reason, not_before):
        if any(r.dedupe_key == event.dedupe_key for r in self.rows.values()):
            return None
        return self.add(event, now, status=status, reason=reason, not_before=not_before)

    def foldable(self, since, until):
        return sorted((r for r in self.rows.values() if r.status == "queued" and not r.owed and not r.brief
                       and r.urgency == "routine" and r.created_at and r.created_at >= since
                       and (r.not_before is None or r.not_before <= until)), key=lambda r: r.id)

    def sent_quietly(self, since):
        return sorted((r for r in self.rows.values() if r.status == "sent" and r.silent
                       and r.delivered_at and r.delivered_at >= since), key=lambda r: r.id)

    def sent_since(self, since):
        return sum(1 for r in self.rows.values() if r.status == "sent" and not r.owed and not r.brief
                   and r.urgency != "safety" and r.delivered_at and r.delivered_at >= since)

    def due(self, now):
        return sorted((r for r in self.rows.values() if r.status == "queued"
                       and (r.not_before is None or r.not_before <= now)), key=lambda r: r.id)

    def awaiting_reply(self, before, after, detail_after=None):
        lower = lambda r: detail_after if (r.detail and detail_after is not None) else after
        return [r for r in self.rows.values() if r.status == "sent" and r.spoken_at and r.dm_at is None
                and r.acked_at is None and (not r.brief or r.detail) and lower(r) <= r.spoken_at <= before]

    def mark(self, reach_id, now, **fields):
        bad = set(fields) - MARKABLE
        if bad:
            raise ValueError(f"not markable: {bad}")
        for k, v in fields.items():
            setattr(self.rows[reach_id], k, v)

    def settle(self, reach_id, now, *, folds=(), **fields):
        self.mark(reach_id, now, **fields)
        for fid in folds:                    # only what's still waiting: never re-mark what went
            r = self.rows.get(fid)
            if r is not None and r.status == "queued":
                r.status, r.reason, r.delivered_at = "folded", f"in the brief (#{reach_id})", now

    def prefs(self, default_budget):
        if self._prefs is None:
            self._prefs = Prefs(budget=default_budget)
        return replace(self._prefs, muted=list(self._prefs.muted))

    def set_prefs(self, prefs, now, change, words):
        self._prefs = replace(prefs, muted=list(prefs.muted))
        self.changes.append((now, change, words))


class PostgresReachStore:
    """v2_reachouts / v2_reach_prefs / v2_reach_pref_changes (db/migrations/016). Fresh short
    connections per call, the tickets store's stance. Times are the brain's clock."""

    COLUMNS = ("id, source, topic, urgency, owed, headline, detail, dedupe_key, status, not_before, expires_at, "
               "reason, surface, silent, spoken_at, dm_at, delivered_at, acked_at, attempts, created_at, "
               "room, brief, folds, text_only, text_to")

    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url, connect_timeout=5, options="-c statement_timeout=5000")

    @staticmethod
    def _row(r) -> Reach:
        return Reach(id=r[0], source=r[1], topic=r[2], urgency=r[3], owed=r[4], headline=r[5], detail=r[6],
                     dedupe_key=r[7], status=r[8], not_before=r[9], expires_at=r[10], reason=r[11], surface=r[12],
                     silent=r[13], spoken_at=r[14], dm_at=r[15], delivered_at=r[16], acked_at=r[17],
                     attempts=r[18], created_at=r[19], room=r[20], brief=r[21], folds=tuple(r[22] or ()), text_only=bool(r[23]),
                     text_to=r[24])

    def add(self, event, now, *, status, reason, not_before):
        with self._connect() as conn:
            row = conn.execute(
                "INSERT INTO v2_reachouts (created_at, updated_at, source, topic, urgency, owed, headline, detail, "
                "dedupe_key, status, not_before, expires_at, reason, room, brief, folds, text_only, text_to) VALUES "
                f"(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING {self.COLUMNS}",
                (now, now, event.source, event.topic, event.urgency, event.owed, event.headline, event.detail,
                 event.dedupe_key, status, not_before, event.expires_at, reason, event.room, event.brief,
                 list(event.folds), event.text_only, event.text_to)).fetchone()
        return self._row(row)

    def add_new(self, event, now, *, status, reason, not_before):
        """add(), unless a row with this dedupe key exists in ANY state, ever: one statement,
        for a producer outside the brain (the email watcher) whose key names one thing for
        good. None when it's already there."""
        with self._connect() as conn:
            row = conn.execute(
                "INSERT INTO v2_reachouts (created_at, updated_at, source, topic, urgency, owed, headline, detail, "
                "dedupe_key, status, not_before, expires_at, reason, room, brief, folds, text_only, text_to) "
                "SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s "
                "WHERE NOT EXISTS (SELECT 1 FROM v2_reachouts WHERE dedupe_key = %s) "
                # Two producers at once: the unique index (migration 019) turns the loser away.
                f"ON CONFLICT DO NOTHING RETURNING {self.COLUMNS}",
                (now, now, event.source, event.topic, event.urgency, event.owed, event.headline, event.detail,
                 event.dedupe_key, status, not_before, event.expires_at, reason, event.room, event.brief,
                 list(event.folds), event.text_only, event.text_to, event.dedupe_key)).fetchone()
        return self._row(row) if row else None

    def recently_sent(self, dedupe_key, since):
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM v2_reachouts WHERE dedupe_key = %s AND status IN ('sent', 'queued') "
                "AND created_at >= %s LIMIT 1", (dedupe_key, since)).fetchone() is not None

    def decided(self, dedupe_key, since):
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM v2_reachouts WHERE dedupe_key = %s AND created_at >= %s LIMIT 1",
                (dedupe_key, since)).fetchone() is not None

    def withdraw(self, dedupe_key, now, reason="he called it off"):
        with self._connect() as conn:
            return conn.execute(
                "UPDATE v2_reachouts SET status = 'dropped', reason = %s, updated_at = %s "
                "WHERE dedupe_key = %s AND status = 'queued' RETURNING id",
                (reason, now, dedupe_key)).fetchone() is not None

    def statuses(self, dedupe_key):
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT status FROM v2_reachouts WHERE dedupe_key = %s",
                                (dedupe_key,)).fetchall()
        return {r[0] for r in rows}

    def carried(self, dedupe_key, since):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT w.status FROM v2_reachouts m JOIN v2_reachouts w ON m.id = ANY(w.folds) "
                "WHERE m.dedupe_key = %s AND m.status = 'folded' AND m.created_at >= %s "
                "AND w.status IN ('queued', 'sent')", (dedupe_key, since)).fetchall()
        states = {r[0] for r in rows}
        return "sent" if "sent" in states else "pending" if states else None

    def foldable(self, since, until):
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {self.COLUMNS} FROM v2_reachouts WHERE status = 'queued' AND NOT owed AND NOT brief "
                "AND urgency = 'routine' AND created_at >= %s AND (not_before IS NULL OR not_before <= %s) "
                "ORDER BY id", (since, until)).fetchall()
        return [self._row(r) for r in rows]

    def sent_quietly(self, since):
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {self.COLUMNS} FROM v2_reachouts WHERE status = 'sent' AND silent "
                "AND delivered_at >= %s ORDER BY id", (since,)).fetchall()
        return [self._row(r) for r in rows]

    def sent_since(self, since):
        with self._connect() as conn:
            return conn.execute(
                "SELECT count(*) FROM v2_reachouts WHERE status = 'sent' AND NOT owed AND NOT brief "
                "AND urgency <> 'safety' AND delivered_at >= %s", (since,)).fetchone()[0]

    def due(self, now):
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {self.COLUMNS} FROM v2_reachouts WHERE status = 'queued' "
                "AND (not_before IS NULL OR not_before <= %s) ORDER BY id", (now,)).fetchall()
        return [self._row(r) for r in rows]

    def awaiting_reply(self, before, after, detail_after=None):
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {self.COLUMNS} FROM v2_reachouts WHERE status = 'sent' AND spoken_at IS NOT NULL "
                "AND dm_at IS NULL AND acked_at IS NULL AND (NOT brief OR detail <> '') AND spoken_at <= %s "
                "AND spoken_at >= CASE WHEN detail <> '' THEN %s ELSE %s END ORDER BY id",
                (before, detail_after or after, after)).fetchall()
        return [self._row(r) for r in rows]

    def mark(self, reach_id, now, **fields):
        bad = set(fields) - MARKABLE
        if bad:
            raise ValueError(f"not markable: {bad}")
        cols = ", ".join(f"{k} = %s" for k in fields)
        with self._connect() as conn:
            conn.execute(f"UPDATE v2_reachouts SET {cols}, updated_at = %s WHERE id = %s",
                         (*fields.values(), now, reach_id))

    def settle(self, reach_id, now, *, folds=(), **fields):
        """The record and the items it carried, in ONE transaction: a brief can't be recorded
        as sent while what it carried still waits to go out again on its own."""
        bad = set(fields) - MARKABLE
        if bad:
            raise ValueError(f"not markable: {bad}")
        cols = ", ".join(f"{k} = %s" for k in fields)
        with self._connect() as conn:
            if fields:
                conn.execute(f"UPDATE v2_reachouts SET {cols}, updated_at = %s WHERE id = %s",
                             (*fields.values(), now, reach_id))
            if folds:
                conn.execute("UPDATE v2_reachouts SET status = 'folded', reason = %s, delivered_at = %s, "
                             "updated_at = %s WHERE id = ANY(%s) AND status = 'queued'",
                             (f"in the brief (#{reach_id})", now, now, list(folds)))

    def prefs(self, default_budget):
        with self._connect() as conn:
            row = conn.execute("SELECT budget, muted, hold_since, hold_until, checkins, company_until "
                               "FROM v2_reach_prefs WHERE id = 1").fetchone()
        if not row:
            return Prefs(budget=default_budget)
        return Prefs(budget=row[0], muted=list(row[1]), hold_since=row[2], hold_until=row[3], checkins=row[4],
                     company_until=row[5])

    def set_prefs(self, prefs, now, change, words):
        with self._connect() as conn:   # one transaction: the prefs and their record together
            conn.execute(
                "INSERT INTO v2_reach_prefs (id, budget, muted, hold_since, hold_until, checkins, company_until, "
                "updated_at) VALUES (1, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
                "budget = EXCLUDED.budget, muted = EXCLUDED.muted, hold_since = EXCLUDED.hold_since, "
                "hold_until = EXCLUDED.hold_until, checkins = EXCLUDED.checkins, "
                "company_until = EXCLUDED.company_until, updated_at = EXCLUDED.updated_at",
                (prefs.budget, list(prefs.muted), prefs.hold_since, prefs.hold_until, prefs.checkins,
                 prefs.company_until, now))
            conn.execute("INSERT INTO v2_reach_pref_changes (at, change, words) VALUES (%s, %s, %s)",
                         (now, change, words))


# --- delivery ----------------------------------------------------------------------------


class AmbiguousDelivery(Exception):
    """A leg sent it but couldn't hear back (a timeout after the request went): it may have landed,
    so nothing is sent a second time."""


@dataclass
class Where:
    kind: str                         # room | device | notify (he's away: phone, watch, glasses) | discord | telegram
    device_id: str | None = None
    label: str = ""


@dataclass
class Delivery:
    status: str                       # sent | busy | wait | away | recheck | failed
    surface: str = ""
    spoken: bool = False
    texted: bool = False


class Deliverer:
    """Where he is, and how to reach him there. Every leg is injected (tests; production
    wires them in reach_desk_for)."""

    def __init__(self, *, locate: Callable[[], Where], speak: Callable[[str, str], None],
                 text: Callable[[str, str, bool], None], busy: Callable[[str], bool] | None = None,
                 text_surface: str = "discord", notify: Callable[[str, str], None] | None = None):
        self.locate, self.speak, self.text = locate, speak, text
        self.notify = notify          # (headline, body) -> an HA notification: phone, watch, glasses
        self.busy = busy or (lambda device_id: False)
        self.text_surface = text_surface
        # Set by the desk: may she speak this RIGHT NOW? Checked at the moment of speech, after
        # the location and busy checks, because the night or a meeting can begin while those run.
        self.may_speak: Callable[[Reach], bool] = lambda row: True
        # Set by the desk: may a safety warning (which no hold stops) be SAID right now, or
        # only texted? Checked at the moment of speech too: a warning decided at 20:59 must
        # not be spoken at 21:00 unless the owner turned night voice on.
        self.safety_voice: Callable[[Reach], bool] = lambda row: True

    def deliver(self, row: Reach, decision: Decision) -> Delivery:
        try:
            # A text_only message goes to its own surface when it names one (D20: a ticket asked on
            # Telegram is answered on Telegram), else the desk's default text surface.
            where = Where(row.text_to or self.text_surface) if row.text_only else self.locate()
        except Exception as e:      # can't place him: text (a pinned brief then waits for its room)
            log.warning("reach: couldn't place him (%s); texting", type(e).__name__)
            where = Where(self.text_surface)
        if row.room and not (where.kind == "room" and where.label == row.room):
            return Delivery("away")        # pinned to a room he isn't in: it waits for him there
        body = row.headline + (f"\n\n{row.detail}" if row.detail else "")
        # A notification on his phone, watch and glasses, AS WELL AS the text (the owner, 11/08: "and
        # not instead of, she can discord me too"): in company (never aloud) and when he's away ("if im
        # not home she can use that channel too"). Not for a quiet one: a notification buzzes.
        if (decision.voice == "notify" or where.kind == "notify") and not row.text_only:   # email stays a text
            notified = False
            if self.notify is not None and not decision.silent:
                try:
                    self.notify(row.headline, body)
                    notified = True
                except Exception as e:      # ambiguous or not: the text below carries it either way
                    log.warning("reach: the notification didn't confirm (%s); the text still goes", type(e).__name__)
            try:
                self.text(self.text_surface, body, decision.silent)
                return Delivery("sent", ("notify+" if notified else "") + self.text_surface, texted=True)
            except Exception as e:
                if notified:
                    return Delivery("sent", "notify")        # it reached him; the text didn't (no dm_at)
                log.warning("reach: delivery via %s failed (%s)", self.text_surface, type(e).__name__)
                return Delivery("failed", self.text_surface)
        spoken = where.kind in ("room", "device")
        if spoken and decision.voice == "wait":
            return Delivery("wait")
        if spoken and decision.voice == "reroute":
            where, spoken = Where(self.text_surface), False
        try:
            if spoken:
                if self.busy(where.device_id):
                    if decision.breaks_through:      # safety never waits on a room mid-conversation
                        return self._text_now(body, decision)
                    return Delivery("busy")
                if decision.breaks_through and not self.safety_voice(row):   # the night began just now
                    return self._text_now(body, decision)
                if not decision.breaks_through and not self.may_speak(row):   # the night or a meeting began just now
                    return Delivery("recheck")
                self.speak(row.headline, where.device_id)
            else:
                self.text(where.kind, body, decision.silent)
                return Delivery("sent", where.kind, texted=True)
        except Exception as e:
            log.warning("reach: delivery via %s failed (%s)", where.kind, type(e).__name__)
            if spoken and decision.breaks_through:   # the speaker failed: safety goes as text, now
                return self._text_now(body, decision)
            return Delivery("failed", where.kind)
        # Spoken. The detail goes as text; if THAT fails the message still reached him (it is
        # never spoken twice): the watcher's follow-through sends the detail later.
        surface = f"{where.kind}:{where.label or where.device_id}"
        if not row.detail:
            return Delivery("sent", surface, spoken=True)
        try:
            self.text(self.text_surface, body, decision.silent)
            return Delivery("sent", surface, spoken=True, texted=True)
        except Exception as e:
            log.warning("reach: spoken, but the detail text failed (%s); the follow-through sends it", type(e).__name__)
            return Delivery("sent", surface, spoken=True)


    def _text_now(self, body: str, decision: Decision) -> Delivery:
        """Safety that couldn't be said: one text attempt (a retry is the desk's, never a
        second send here: a text whose answer timed out may have landed)."""
        try:
            self.text(self.text_surface, body, decision.silent)
            return Delivery("sent", self.text_surface, texted=True)
        except Exception as e:
            log.warning("reach: safety couldn't be said, and the text failed too (%s)", type(e).__name__)
            return Delivery("failed", self.text_surface)


# --- the desk ----------------------------------------------------------------------------


class ReachDesk:
    """reach_out(event) -> 'sent' | 'queued' | 'dropped' | 'duplicate'. 'queued' is a promise: the watcher
    delivers it when its window opens (or retries a failure), and an answer he asked for is
    never dropped. said(row, text) records in her thread what she actually said."""

    def __init__(self, store: ReachStore, deliverer: Deliverer, *, budget_default: int = 3,
                 quiet: Callable[[datetime], bool] | None = None, morning_hour: int = 9,
                 meeting_hold: Callable[[], datetime | None] | None = None,
                 replied_since: Callable[[datetime], bool] | None = None,
                 said: Callable[[Reach, str], None] | None = None,
                 follow_through_after: timedelta = timedelta(minutes=5),
                 safety_voice_at_night: bool = False,
                 clock: Callable[[], datetime] | None = None):
        self.store, self.deliverer = store, deliverer
        self.safety_voice_at_night = safety_voice_at_night
        self.budget_default, self.morning_hour = budget_default, morning_hour
        self.quiet = quiet or (lambda now: False)
        self.meeting_hold = meeting_hold
        self.replied_since = replied_since
        self.said = said or (lambda row, text: None)
        self.follow_through_after = follow_through_after
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.lock = threading.Lock()   # one delivery at a time: the watcher and a turn can race
        self.wake = threading.Event()
        # Rows delivered whose record didn't land (the store hiccuped after the send): only
        # the record is retried, never the delivery (the check-ins' stance; Codex 11/08).
        self.unrecorded: dict = {}
        deliverer.may_speak = self._may_speak
        deliverer.safety_voice = lambda row: self.safety_voice_at_night or not self.quiet(self.clock())

    def _may_speak(self, row: Reach) -> bool:
        """May she say this RIGHT NOW (the deliverer asks at the moment of speech)? Never past
        its expiry (a brief composed at 12:59 doesn't speak at 13:01, nor at night); the night
        holds everything but a brief; a meeting or a nap holds everything."""
        now = self.clock()
        if row.expires_at is not None and not row.owed and now >= _utc(row.expires_at):
            return False
        prefs = self.store.prefs(self.budget_default)
        company = prefs.company_until is not None and _utc(prefs.company_until) > now
        return ((row.brief or not self.quiet(now)) and self._meeting() is None and not company
                and self._nap(prefs, now) is None)

    def _meeting(self) -> datetime | None:
        try:
            return self.meeting_hold() if self.meeting_hold else None
        except Exception:
            log.warning("reach: meeting check failed; not holding", exc_info=True)
            return None

    def _nap(self, prefs: Prefs, now: datetime) -> datetime | None:
        """A nap he told her about: it holds until hold_until, or until he talks to her again
        (a turn of his after NAP_GRACE, so the turn that started the nap doesn't end it)."""
        if prefs.hold_until is None or _utc(prefs.hold_until) <= now:
            return None
        if self.replied_since is not None and prefs.hold_since is not None:
            try:
                if self.replied_since(_utc(prefs.hold_since) + NAP_GRACE):
                    return None
            except Exception:
                log.warning("reach: nap check failed; still holding", exc_info=True)
        return _utc(prefs.hold_until)

    def _decide(self, event: ReachEvent, now: datetime, *, fresh: bool) -> Decision:
        duplicate = bool(fresh and event.dedupe_key
                         and self.store.recently_sent(event.dedupe_key, now - DEDUPE_WINDOW))
        prefs = self.store.prefs(self.budget_default)
        company = prefs.company_until is not None and _utc(prefs.company_until) > now
        meeting, nap = self._meeting(), self._nap(prefs, now)
        hold = "nap" if nap is not None and (meeting is None or nap >= _utc(meeting)) else "meeting"
        until = max(t for t in (_utc(meeting), nap) if t is not None) if (meeting or nap) else None
        return decide(event, now, prefs=prefs, sent_today=self.store.sent_since(local_day_start(now)),
                      duplicate=duplicate, quiet=self.quiet(now), meeting_until=until,
                      morning_hour=self.morning_hour, hold=hold,
                      safety_voice_at_night=self.safety_voice_at_night, company=company)

    def voice_held(self, now: datetime | None = None) -> str | None:
        """Why her voice is held right now (a meeting or a nap), or None. The brief asks before
        it composes, so a held morning doesn't spend a model call on words that would wait."""
        now = now or self.clock()
        if self._meeting() is not None:
            return "meeting"
        if self._nap(self.store.prefs(self.budget_default), now) is not None:
            return "nap"
        return None

    def reserve_for_brief(self, since: datetime, until: datetime, *, reason: str = "carried by the morning brief",
                          skip_topics: tuple = ()) -> list[Reach]:
        """The routine items held since `since` that a brief may carry, reserved for it: each
        waits until `until` (the brief's cutoff), so the watcher can't also send it while she
        composes. Not what has expired or what he's muted since (those the watcher drops). If
        the brief never goes, they go on their own at `until`."""
        with self.lock:
            now = self.clock()
            muted = muted_topics(self.store.prefs(self.budget_default))
            out = []
            for r in self.store.foldable(since, until):
                if self._delivered(r.id) or r.topic.strip().lower() in muted or r.topic in skip_topics:
                    continue
                if r.expires_at is not None and now >= _utc(r.expires_at):
                    continue
                try:
                    self.store.mark(r.id, now, not_before=until, reason=reason)
                except Exception:
                    log.warning("reach %s: couldn't reserve it for the brief; it goes on its own", r.id)
                    continue
                out.append(r)
            return out

    def take_back(self, dedupe_key: str, now: datetime, reason: str) -> str:
        """Something else will say what's under this key (the welcome-home hello asks an appointment's
        question itself). Read and, if it's still waiting, withdrawn in ONE step under the delivery
        lock, so it can't be said in between: 'free' (nothing was said or waits), 'taken' (it
        waited and is withdrawn), or 'went' (it was said, or a delivery waiting for its record may
        be it): then it isn't the other's to say."""
        with self.lock:
            states = self.store.statuses(dedupe_key)
            if states & {"sent", "folded"} or ("queued" in states and self.unrecorded):
                return "went"            # said, or carried by a brief that was said
            if "queued" in states:
                return "taken" if self.store.withdraw(dedupe_key, now, reason=reason) else "went"
            return "free"

    def _delivered(self, rid: int) -> bool:
        """Delivered (or carried by a delivered brief) but not yet recorded: never send it again."""
        return rid in self.unrecorded or any(rid in folds for _, folds in self.unrecorded.values())

    def _settle(self, rid: int, now: datetime, record: dict, folds: tuple = ()) -> None:
        """Record a delivery (and what it carried). Noted first in `unrecorded`, so a failed
        write is retried by the watcher and the delivery itself is never repeated."""
        self.unrecorded[rid] = (record, folds)
        try:
            self.store.settle(rid, now, folds=folds, **record)
            self.unrecorded.pop(rid, None)
        except Exception:
            log.warning("reach %s: delivered, but recording it failed; only the record is retried", rid)

    def reach_out(self, event: ReachEvent) -> str:
        if event.urgency not in URGENCIES or not event.headline.strip():
            raise ValueError("a reach-out needs a headline and a known urgency")
        with self.lock:
            now = self.clock()
            d = self._decide(event, now, fresh=True)
            if d.action == "drop":
                row = self.store.add(event, now, status="dropped", reason=d.reason, not_before=None)
                log.info("reach %s (%s): dropped: %s", row.id, event.source, d.reason)
                # "duplicate" = the desk already holds this one (sent, or queued for its window):
                # a caller retrying the same item (a ticket relay whose own record failed) has
                # been heard, not refused.
                return "duplicate" if d.reason == "already sent" else "dropped"
            # Written BEFORE any attempt: a crash mid-delivery leaves it queued, never lost.
            row = self.store.add(event, now, status="queued", reason=d.reason,
                                 not_before=d.not_before if d.action == "queue" else None)
            if d.action == "queue":
                log.info("reach %s (%s): queued: %s", row.id, event.source, d.reason)
                return "queued"
            return self._attempt(row, d, now)

    def _attempt(self, row: Reach, d: Decision, now: datetime, *, rechecked: bool = False) -> str:
        result = self.deliverer.deliver(row, d)
        if result.status == "recheck":
            # Speech was allowed when decided, not any more: decide again with the fresh clock.
            now = self.clock()
            if rechecked:
                self.store.mark(row.id, now, not_before=now + BUSY_RETRY, reason="the window changed; retrying")
                return "queued"
            d2 = self._decide(row.event(), now, fresh=False)
            if d2.action == "drop":
                self.store.mark(row.id, now, status="dropped", reason=d2.reason)
                return "dropped"
            if d2.action == "queue":
                self.store.mark(row.id, now, not_before=d2.not_before, reason=d2.reason)
                return "queued"
            return self._attempt(row, d2, now, rechecked=True)
        if result.status == "sent":
            record = dict(status="sent", delivered_at=now, surface=result.surface, silent=d.silent,
                          spoken_at=now if result.spoken else None, dm_at=now if result.texted else None,
                          attempts=row.attempts + 1)
            self._settle(row.id, now, record, tuple(row.folds))   # the items it carried went with it
            try:
                self.said(row, row.headline + (f"\n\n{row.detail}" if row.detail else ""))
            except Exception:
                log.warning("reach %s: delivered, but her thread didn't take the note", row.id, exc_info=True)
            log.info("reach %s (%s): sent via %s%s", row.id, row.source, result.surface, " (quiet)" if d.silent else "")
            return "sent"
        if result.status == "wait":
            self.store.mark(row.id, now, not_before=d.not_before, reason="meeting: waits for the window")
            return "queued"
        if result.status == "busy":
            self.store.mark(row.id, now, not_before=now + BUSY_RETRY, reason="the room was mid-conversation")
            return "queued"
        if result.status == "away":
            self.store.mark(row.id, now, not_before=now + ROOM_WAIT, reason=f"waits for him in the {row.room}")
            return "queued"
        attempts = row.attempts + 1
        if attempts >= GIVE_UP_AFTER and not row.owed:
            self.store.mark(row.id, now, status="failed", attempts=attempts, reason="couldn't reach you")
            log.warning("reach %s (%s): gave up after %d attempts", row.id, row.source, attempts)
            return "dropped"
        backoff = FAIL_BACKOFF[min(attempts - 1, len(FAIL_BACKOFF) - 1)]
        self.store.mark(row.id, now, not_before=now + backoff, attempts=attempts,
                        reason=f"delivery failed; retrying in {int(backoff.total_seconds() // 60)} min")
        return "queued"

    def tick(self) -> dict:
        """The watcher's pass: deliver what's due, and text a copy of what he didn't answer."""
        out = {"sent": 0, "queued": 0, "dropped": 0, "duplicate": 0, "copied": 0, "heard": 0}
        with self.lock:
            now = self.clock()
            for rid, (record, folds) in list(self.unrecorded.items()):   # delivered; only the record is owed
                try:
                    self.store.settle(rid, now, folds=folds, **record)
                    self.unrecorded.pop(rid, None)
                except Exception:
                    log.warning("reach %s: the record still won't land; retried next pass", rid)
            for row in self.store.due(now):
                if self._delivered(row.id):            # sent already (or carried): never delivered twice
                    continue
                now = self.clock()                     # per row: a slow delivery can cross 21:00
                if row.expires_at and _utc(row.expires_at) <= now and not row.owed:
                    self.store.mark(row.id, now, status="dropped", reason="expired before its window")
                    out["dropped"] += 1
                    continue
                d = self._decide(row.event(), now, fresh=False)
                if d.action == "drop":
                    self.store.mark(row.id, now, status="dropped", reason=d.reason)
                    out["dropped"] += 1
                elif d.action == "queue":
                    self.store.mark(row.id, now, not_before=d.not_before, reason=d.reason)
                    out["queued"] += 1
                else:
                    out[self._attempt(row, d, now)] += 1
            if self.replied_since is not None:
                for row in self.store.awaiting_reply(now - self.follow_through_after, now - FOLLOW_THROUGH_FOR,
                                                     detail_after=now - DETAIL_FOR):
                    if self._delivered(row.id):
                        continue
                    try:
                        # A brief is never re-sent as "in case you missed it" (his phone has
                        # Kael's); only its text part, if that didn't land.
                        heard = True if row.brief else self.replied_since(_utc(row.spoken_at))
                    except Exception:
                        log.warning("reach: reply check failed; no copy this pass", exc_info=True)
                        continue
                    if heard and not row.detail:
                        self.store.mark(row.id, now, acked_at=now)
                        out["heard"] += 1
                        continue
                    # Not answered: a copy. Answered but the detail never went (its text failed
                    # after the speech): the detail alone, he heard the headline.
                    copy = (row.detail if heard else
                            f"{MISSED_PREFIX} {row.headline}" + (f"\n\n{row.detail}" if row.detail else ""))
                    quiet_copy = self.quiet(now) and row.urgency != "safety"   # a warning still buzzes
                    try:
                        self.deliverer.text(self.deliverer.text_surface, copy, quiet_copy)
                    except Exception:
                        log.warning("reach %s: the text copy failed; retried next pass", row.id)
                        continue
                    # He doesn't get Discord notifications (11/08): a copy of something he missed also
                    # pings his phone, watch and glasses, unless it's meant to be quiet.
                    if not heard and not quiet_copy and self.deliverer.notify is not None:
                        try:
                            self.deliverer.notify(row.headline, copy)
                        except Exception:
                            log.warning("reach %s: the copy's notification didn't confirm (the text went)", row.id)
                    # first noted: a failed record must not mean a second copy
                    self._settle(row.id, now, {"dm_at": now, **({"acked_at": now} if heard else {})})
                    out["copied"] += 1
        return out

    def run_forever(self) -> None:  # pragma: no cover - thin loop over tick()
        log.info("reach-out desk up | every %d s", TICK_S)
        while True:
            try:
                self.tick()
            except Exception:
                log.warning("reach desk tick failed", exc_info=True)
            self.wake.wait(TICK_S)
            self.wake.clear()


def start_reach_desk(desk: "ReachDesk | None") -> threading.Thread | None:
    if desk is None:
        return None
    thread = threading.Thread(target=desk.run_forever, daemon=True, name="reach-desk")
    thread.start()
    return thread
