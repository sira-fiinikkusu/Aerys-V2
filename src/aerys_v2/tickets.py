"""Kael tickets: what she asks Kael comes back, and what the owner asked comes back to him.

The owner, 2026-11-07: "when i tell her to ask you something, she never follows up with
me. the conversation is 1 way." Her line to Kael (message_kael) was fire-and-forget: the
words landed in his session, his answer (if he remembered to send one) landed in her
thread, and nothing ever brought it back to the person who asked. Board #176, Stage 0.

So every message_kael opens a ticket (db/migrations/015). The id rides in the message
("(ticket T-12) ..."), and Kael answers BY TICKET through the brain's door
(POST /kael-ticket/{id}). The answer is injected into the thread she asked from (the same
note injection as /kael-note), and when the owner was the one asking, the watcher here has
her bring it to him: no HOLD option, quiet hours make it a silent message instead of a
spoken one. Statuses are A2A task states, and her verbs (message_kael starts one;
kael_tickets checks, updates, cancels and lists) match the async-task verbs, so Kael could
later sit behind the same tools as her home body without a rebuild.

The watcher (one daemon in --serve, like the check-ins) does four things each minute:
  * resend: an ask the line didn't accept is resent every RESEND_EVERY while it is fresh
    (Kael's line only exists while his session runs);
  * nag: a delivered ask with no answer after the nag delay reminds Kael once;
  * tell the owner: an ask HE made with no answer after the owner delay is reported to him
    once ("still open" or "couldn't reach Kael");
  * relay: Kael's answer to an ask the owner made is brought to him, at least once.
The delays are settings (D15 on #176 is the owner's call; these are the proposed defaults).
Delivery is at-least-once, the same stance as the check-ins: an answer that was relayed but
couldn't be recorded is remembered in the watcher and never relayed again by it.

The ask and the answer are the private part: never logged. Logs carry ids and statuses.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Protocol

import httpx

log = logging.getLogger(__name__)

OPEN = ("submitted", "working", "input-required")
WAITING_ON_KAEL = ("submitted", "working")
KAEL_STATUSES = ("working", "input-required", "completed", "failed", "canceled")
#: Kael's words on these come back to the owner; "working" is progress she only notes.
OWED_TO_OWNER = ("input-required", "completed", "failed")

#: The desk channel shows up to 4000 characters of a ping. Text still owed to the line is
#: bounded below that, so a ticket's whole outbox always fits in one send (nothing is cut).
LINE_LIMIT = 4000
OUTBOX_LIMIT = 3800
UPDATE_LIMIT = 1500
TICK_S = 60
RESEND_EVERY = timedelta(minutes=10)
RESEND_FOR = timedelta(hours=72)
RELAY_FOR = timedelta(hours=24)


def label(ticket_id: int) -> str:
    return f"T-{ticket_id}"


_ID_RE = re.compile(r"^\s*(?:#|t-?|ticket\s*)?(\d{1,12})\s*$", re.IGNORECASE)


def parse_id(text) -> int | None:
    """'T-12', 't12', '#12', 'ticket 12' and '12' all mean ticket 12."""
    m = _ID_RE.match(str(text or ""))
    return int(m.group(1)) if m else None


def _utc(dt: datetime | None) -> datetime | None:
    return dt.astimezone(timezone.utc) if dt is not None else None


@dataclass
class Ticket:
    id: int
    thread_id: str
    ask: str
    for_owner: bool = False
    status: str = "submitted"
    created_at: datetime | None = None
    outbox: str | None = None          # text still owed to Kael's line
    last_try_at: datetime | None = None
    delivered_at: datetime | None = None
    waiting_since: datetime | None = None
    answer: str | None = None          # words owed back to the owner (OWED_TO_OWNER statuses)
    answered_at: datetime | None = None
    progress: str | None = None        # words while working: never replaces an answer
    relayed_at: datetime | None = None
    nagged_at: datetime | None = None
    escalated_at: datetime | None = None

    @property
    def relay_owed(self) -> bool:
        return bool(self.for_owner and self.answered_at is not None
                    and (self.relayed_at is None or self.relayed_at < self.answered_at))

    def public(self) -> dict:
        """The door's JSON shape (Kael's authed side): everything, timestamps as ISO."""
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.isoformat() if isinstance(v, datetime) else v
        out["label"] = label(self.id)
        return out


class TicketStore(Protocol):
    def open(self, thread_id: str, ask: str, for_owner: bool, now: datetime) -> Ticket: ...
    def get(self, ticket_id: int) -> Ticket | None: ...
    def recent(self, limit: int = 10, open_only: bool = False) -> list[Ticket]: ...
    def pending(self, now: datetime) -> list[Ticket]: ...
    def tried(self, ticket_id: int, now: datetime) -> None: ...
    def delivered(self, ticket_id: int, now: datetime, sent: str) -> bool: ...
    def from_kael(self, ticket_id: int, status: str, text: str, now: datetime) -> Ticket | None: ...
    def from_her(self, ticket_id: int, text: str, now: datetime) -> tuple[Ticket | None, bool]: ...
    def cancel(self, ticket_id: int, now: datetime) -> Ticket | None: ...
    def mark(self, ticket_id: int, column: str, now: datetime) -> None: ...


MARKS = ("relayed_at", "nagged_at", "escalated_at")


class MemoryTicketStore:
    """In-process store (tests, dev boxes). Same semantics as the Postgres one."""

    def __init__(self, items: Iterable[Ticket] = ()):
        self.items = {t.id: t for t in items}
        self._next = max(self.items, default=0) + 1

    def open(self, thread_id, ask, for_owner, now):
        t = Ticket(id=self._next, thread_id=thread_id, ask=ask, for_owner=for_owner,
                   created_at=now, outbox=ask, last_try_at=now, waiting_since=now)
        self.items[t.id] = t
        self._next += 1
        return t

    def get(self, ticket_id):
        return self.items.get(ticket_id)

    def recent(self, limit=10, open_only=False):
        rows = [t for t in self.items.values() if not open_only or t.status in OPEN]
        return sorted(rows, key=lambda t: t.id, reverse=True)[:limit]

    def pending(self, now):
        return [t for t in self.items.values()
                if t.status in OPEN or (t.relay_owed and now - t.answered_at < RELAY_FOR)]

    def tried(self, ticket_id, now):
        self.items[ticket_id].last_try_at = now

    def delivered(self, ticket_id, now, sent):
        t = self.items[ticket_id]
        t.delivered_at, t.last_try_at = now, now
        if t.outbox != sent:
            return False       # more was queued while this send was out: keep it owed
        t.outbox = None
        return True

    def from_kael(self, ticket_id, status, text, now):
        t = self.items.get(ticket_id)
        if t is None:
            return None
        t.status = status
        if text and status in OWED_TO_OWNER:
            t.answer, t.answered_at = text, now
        elif text:
            t.progress = text
        if status not in OPEN:
            t.outbox = None    # closed: nothing more goes down the line for it
        return t

    def from_her(self, ticket_id, text, now):
        """(ticket, accepted): accepted only when the text was appended."""
        t = self.items.get(ticket_id)
        if t is None or t.status not in OPEN:
            return t, False
        if len(t.outbox or "") + 1 + len(text) > OUTBOX_LIMIT:
            return t, False    # wouldn't fit in one send: refused, unchanged (the tool says so)
        t.outbox = f"{t.outbox}\n{text}" if t.outbox else text   # never drop what's still unsent
        t.status, t.last_try_at, t.waiting_since = "submitted", now, now
        return t, True

    def cancel(self, ticket_id, now):
        t = self.items.get(ticket_id)
        if t is not None and t.status in OPEN:
            t.status, t.outbox = "canceled", None
        return t

    def mark(self, ticket_id, column, now):
        if column not in MARKS:
            raise ValueError(column)
        setattr(self.items[ticket_id], column, now)


class PostgresTicketStore:
    """v2_kael_tickets (db/migrations/015). Fresh short connections per call, the same stance
    as the check-ins store. Every timestamp is the brain's clock (passed in), so the watcher
    compares like with like."""

    COLUMNS = ("id, thread_id, ask, for_owner, status, created_at, outbox, last_try_at, "
               "delivered_at, waiting_since, answer, answered_at, relayed_at, nagged_at, escalated_at, "
               "progress")

    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url, connect_timeout=5,
                               options="-c statement_timeout=5000")

    @staticmethod
    def _ticket(r) -> Ticket:
        return Ticket(id=r[0], thread_id=r[1], ask=r[2], for_owner=r[3], status=r[4],
                      created_at=r[5], outbox=r[6], last_try_at=r[7], delivered_at=r[8],
                      waiting_since=r[9], answer=r[10], answered_at=r[11], relayed_at=r[12],
                      nagged_at=r[13], escalated_at=r[14], progress=r[15])

    def _one(self, sql: str, params: tuple) -> Ticket | None:
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return self._ticket(row) if row else None

    def _many(self, sql: str, params: tuple) -> list[Ticket]:
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._ticket(r) for r in rows]

    def open(self, thread_id, ask, for_owner, now):
        return self._one(
            "INSERT INTO v2_kael_tickets (thread_id, ask, for_owner, created_at, updated_at, outbox, "
            "last_try_at, waiting_since) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            f"RETURNING {self.COLUMNS}",
            (thread_id, ask, for_owner, now, now, ask, now, now))

    def get(self, ticket_id):
        return self._one(f"SELECT {self.COLUMNS} FROM v2_kael_tickets WHERE id = %s", (ticket_id,))

    def recent(self, limit=10, open_only=False):
        where = "WHERE status = ANY(%s)" if open_only else ""
        params = (list(OPEN), limit) if open_only else (limit,)
        return self._many(f"SELECT {self.COLUMNS} FROM v2_kael_tickets {where} ORDER BY id DESC LIMIT %s",
                          params)

    def pending(self, now):
        return self._many(
            f"SELECT {self.COLUMNS} FROM v2_kael_tickets WHERE status = ANY(%s) "
            "OR (for_owner AND answered_at IS NOT NULL "
            "    AND (relayed_at IS NULL OR relayed_at < answered_at) AND answered_at > %s) "
            "ORDER BY id",
            (list(OPEN), now - RELAY_FOR))

    def _exec(self, sql: str, params: tuple) -> None:
        with self._connect() as conn:
            conn.execute(sql, params)

    def tried(self, ticket_id, now):
        self._exec("UPDATE v2_kael_tickets SET last_try_at = %s, updated_at = %s WHERE id = %s",
                   (now, now, ticket_id))

    def delivered(self, ticket_id, now, sent):
        # Compare-and-clear: an update queued while this send was out stays owed.
        with self._connect() as conn:
            cleared = conn.execute(
                "UPDATE v2_kael_tickets SET outbox = NULL, delivered_at = %s, last_try_at = %s, "
                "updated_at = %s WHERE id = %s AND outbox = %s", (now, now, now, ticket_id, sent)).rowcount
            if not cleared:
                conn.execute("UPDATE v2_kael_tickets SET delivered_at = %s, last_try_at = %s, "
                             "updated_at = %s WHERE id = %s", (now, now, now, ticket_id))
        return cleared > 0

    def from_kael(self, ticket_id, status, text, now):
        owed = bool(text) and status in OWED_TO_OWNER
        progress = bool(text) and not owed
        return self._one(
            "UPDATE v2_kael_tickets SET status = %s, updated_at = %s, "
            "answer = CASE WHEN %s THEN %s ELSE answer END, "
            "answered_at = CASE WHEN %s THEN %s ELSE answered_at END, "
            "progress = CASE WHEN %s THEN %s ELSE progress END, "
            "outbox = CASE WHEN %s THEN outbox ELSE NULL END "
            f"WHERE id = %s RETURNING {self.COLUMNS}",
            (status, now, owed, text, owed, now, progress, text, status in OPEN, ticket_id))

    def from_her(self, ticket_id, text, now):
        # Appended, never replaced: an ask still unsent stays in front of what's added to it.
        updated = self._one(
            "UPDATE v2_kael_tickets SET status = 'submitted', "
            "outbox = CASE WHEN outbox IS NULL THEN %s ELSE outbox || E'\\n' || %s END, "
            "last_try_at = %s, waiting_since = %s, updated_at = %s "
            "WHERE id = %s AND status = ANY(%s) "
            f"AND length(COALESCE(outbox, '')) + 1 + length(%s) <= %s RETURNING {self.COLUMNS}",
            (text, text, now, now, now, ticket_id, list(OPEN), text, OUTBOX_LIMIT))
        return (updated, True) if updated else (self.get(ticket_id), False)

    def cancel(self, ticket_id, now):
        updated = self._one(
            "UPDATE v2_kael_tickets SET status = 'canceled', outbox = NULL, updated_at = %s "
            f"WHERE id = %s AND status = ANY(%s) RETURNING {self.COLUMNS}",
            (now, ticket_id, list(OPEN)))
        return updated or self.get(ticket_id)

    def mark(self, ticket_id, column, now):
        if column not in MARKS:
            raise ValueError(column)
        self._exec(f"UPDATE v2_kael_tickets SET {column} = %s, updated_at = %s WHERE id = %s",
                   (now, now, ticket_id))


# --- the line --------------------------------------------------------------------------


def line_text(ticket: Ticket, text: str) -> str:
    return f"(ticket {label(ticket.id)}) {' '.join(text.split())}"[:LINE_LIMIT]


def line_sender(url: str, token: str, client: httpx.Client | None = None) -> Callable[[Ticket, str], bool]:
    """send(ticket, text) for the watcher and her tool: her line to Kael, carrying the
    ticket's return address. 202 = delivered; anything else is False (retried later)."""
    http = client or httpx.Client(timeout=10.0)

    def send(ticket: Ticket, text: str) -> bool:
        try:
            resp = http.post(url, json={"message": line_text(ticket, text), "thread_id": ticket.thread_id,
                                        "ticket_id": label(ticket.id)},
                             headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError:
            log.info("ticket %s: Kael's line is down", label(ticket.id))
            return False
        if resp.status_code != 202:
            log.info("ticket %s: not delivered (HTTP %s)", label(ticket.id), resp.status_code)
            return False
        return True

    return send


# --- the watcher -----------------------------------------------------------------------


def run_once(store: TicketStore, now: datetime, *, send: Callable[[Ticket, str], bool],
             nag_after: timedelta, tell_owner_after: timedelta,
             relay: Callable[[Ticket], bool] | None = None,
             tell_owner: Callable[[Ticket, bool], bool] | None = None,
             relayed: set | None = None) -> dict:
    """One tick. `now` must be timezone-aware. `relay(ticket)` brings Kael's answer to the
    owner; `tell_owner(ticket, reached)` reports an ask of his that is still open (reached =
    the line accepted it). Either may be None (no owner surface armed): that step is skipped.
    `relayed` (the watcher's own set of (id, answered_at)) keeps an answer that was relayed
    but couldn't be recorded from being relayed again.

    Returns {"resent", "nagged", "told_owner", "relayed", "failed"} (lists of labels).
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(timezone.utc)
    result: dict = {"resent": [], "nagged": [], "told_owner": [], "relayed": [], "failed": []}
    relayed = set() if relayed is None else relayed
    for t in store.pending(now):
        name = label(t.id)
        waiting = now - (_utc(t.waiting_since) or _utc(t.created_at) or now)
        try:
            if t.status in OPEN and t.outbox and waiting < RESEND_FOR and (
                    t.last_try_at is None or now - _utc(t.last_try_at) >= RESEND_EVERY):
                sent = t.outbox
                if send(t, sent):
                    if store.delivered(t.id, now, sent):
                        t.outbox = None
                    t.delivered_at = now
                    result["resent"].append(name)
                else:
                    store.tried(t.id, now)
            # The nag clock starts when Kael could first see it: an ask that sat 5 h behind a
            # dead line and was just delivered is news to him, not overdue.
            seen_for = now - max(x for x in (_utc(t.waiting_since), _utc(t.delivered_at), now - waiting)
                                 if x is not None)
            if (t.status in WAITING_ON_KAEL and not t.outbox and seen_for >= nag_after
                    and (t.nagged_at is None or _utc(t.nagged_at) < _utc(t.waiting_since))):
                hours = int(waiting.total_seconds() // 3600)
                if send(t, f"still open after {hours} h, no answer yet. The ask: {t.ask}"[:1900]):
                    store.mark(t.id, "nagged_at", now)
                    result["nagged"].append(name)
            if (t.for_owner and tell_owner is not None and t.status in WAITING_ON_KAEL
                    and waiting >= tell_owner_after
                    and (t.escalated_at is None or _utc(t.escalated_at) < _utc(t.waiting_since))):
                if tell_owner(t, t.delivered_at is not None and not t.outbox):
                    store.mark(t.id, "escalated_at", now)
                    result["told_owner"].append(name)
            if t.relay_owed and relay is not None and now - _utc(t.answered_at) < RELAY_FOR:
                key = (t.id, _utc(t.answered_at))
                if key not in relayed:
                    if not relay(t):
                        result["failed"].append(name)
                        continue
                    relayed.add(key)            # first: a failed record must not mean a re-relay
                    result["relayed"].append(name)
                store.mark(t.id, "relayed_at", now)
                relayed.discard(key)
        except Exception:
            log.warning("ticket %s: tick step failed; retried next tick", name, exc_info=True)
            result["failed"].append(name)
    return result


class TicketWatcher:
    def __init__(self, store: TicketStore, send: Callable[[Ticket, str], bool], *,
                 nag_after: timedelta, tell_owner_after: timedelta,
                 relay: Callable[[Ticket], bool] | None = None,
                 tell_owner: Callable[[Ticket, bool], bool] | None = None,
                 wake: threading.Event | None = None, tick_s: int = TICK_S,
                 clock: Callable[[], datetime] | None = None):
        self.store, self.send = store, send
        self.nag_after, self.tell_owner_after = nag_after, tell_owner_after
        self.relay, self.tell_owner = relay, tell_owner
        self.wake = wake or threading.Event()
        self._tick_s = tick_s
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.relayed: set = set()

    def tick(self) -> dict:
        result = run_once(self.store, self._clock(), send=self.send, nag_after=self.nag_after,
                          tell_owner_after=self.tell_owner_after, relay=self.relay,
                          tell_owner=self.tell_owner, relayed=self.relayed)
        for step in ("resent", "nagged", "told_owner", "relayed"):
            for name in result[step]:
                log.info("ticket %s: %s", name, step.replace("_", " "))
        return result

    def run_forever(self) -> None:  # pragma: no cover - thin loop over tick()
        log.info("kael tickets up | every %d s (woken early by Kael's answers)", self._tick_s)
        while True:
            try:
                self.tick()
            except Exception:
                log.warning("ticket tick failed", exc_info=True)
            self.wake.wait(self._tick_s)
            self.wake.clear()


# --- Kael's side of the door -------------------------------------------------------------


@dataclass
class TicketDoor:
    """What the HTTP door needs: the store, and Kael's update (answer, progress, cancel)."""

    store: TicketStore
    update: Callable[[int, str, str, bool], Ticket | None]


def kael_update_for(store: TicketStore, note_fn, wake: threading.Event | None = None,
                    clock: Callable[[], datetime] | None = None):
    """fn(ticket_id, status, text, family_visible) -> the updated ticket, or None if unknown.

    Words from Kael go into the thread she asked from FIRST (the /kael-note injection, minus
    its own proactive turn: the relay below is the follow-up, so there's no second one), then
    the row. A failure between the two means Kael retries and she may see the note twice:
    a duplicated note is milder than a duplicated message to the owner."""
    now_fn = clock or (lambda: datetime.now(timezone.utc))

    def update(ticket_id: int, status: str, text: str = "", family_visible: bool = False):
        if status not in KAEL_STATUSES:
            raise ValueError(f"status must be one of {', '.join(KAEL_STATUSES)}")
        ticket = store.get(ticket_id)
        if ticket is None:
            return None
        words = " ".join((text or "").split())
        if words and note_fn is not None:
            note_fn(ticket.thread_id, f"(ticket {label(ticket.id)}, {status}) {words}",
                    family_visible, proactive=False)
        ticket = store.from_kael(ticket_id, status, words, now_fn())
        if ticket is not None and ticket.relay_owed and wake is not None:
            wake.set()          # the watcher brings it to the owner now, not at the next minute
        log.info("ticket %s: Kael set %s", label(ticket_id), status)
        return ticket

    return update


def ticket_door_for(settings, note_fn, wake: threading.Event | None = None) -> TicketDoor | None:
    if not getattr(settings, "database_url", None):
        return None
    store = PostgresTicketStore(settings.database_url)
    return TicketDoor(store=store, update=kael_update_for(store, note_fn, wake))


def start_tickets(settings, graph, wake: threading.Event | None = None, desk=None) -> threading.Thread | None:
    """Arm-and-forget, --serve only (one watcher, like the check-ins): None unless the
    database and Kael's line are both configured. The owner-facing steps arm only when she
    has a way to reach him (kael_line decides)."""
    if not settings.database_url or not settings.kael_desk_url or settings.kael_desk_token is None:
        return None
    from aerys_v2.kael_line import owner_notice_for, ticket_relay_for

    watcher = TicketWatcher(
        PostgresTicketStore(settings.database_url),
        line_sender(settings.kael_desk_url, settings.kael_desk_token.get_secret_value()),
        nag_after=timedelta(hours=settings.kael_ticket_nag_after_h),
        tell_owner_after=timedelta(hours=settings.kael_ticket_tell_owner_after_h),
        relay=ticket_relay_for(graph, settings, desk=desk),
        tell_owner=owner_notice_for(graph, settings, desk=desk),
        wake=wake,
    )
    thread = threading.Thread(target=watcher.run_forever, daemon=True, name="kael-tickets")
    thread.start()
    return thread
