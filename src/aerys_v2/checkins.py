"""Kael check-ins: she taps Kael at set times, so a recurring job can live with her.

Owner ask (2026-10-12): a recurring job Kael runs for the household (a weekly digest,
say) shouldn't hang on Kael's session crons, which expire every week; it should live
with her, and stopping it should be as easy as telling her. A check-in is a name,
days, a local HH:MM, the message to send, and an on/off switch. When one is due she
sends its message down her line to Kael's session (the same lane as message_kael),
labelled as a check-in; Kael decides what it starts. Rows live in the database, never
in this repo. A NAME is a generic label ("monday-digest"): it shows in her list, in
logs and in the tap. The MESSAGE is the private part and is never logged.

The scheduling rule was drafted by Aerys herself, in her own workspace, during the
first Kael sitting (2026-10-12), reviewed by Kael and ported here:

  * Due: one of its days, at or after its time, not yet fired for that day. Fires at
    most once per scheduled day.
  * Late: if the body was down at the time, it still fires once within LATE_WINDOW
    (2 h) of the scheduled time; past that, that day is skipped and reported once.
  * A failed send (raises, or returns False) is not marked fired, so the next tick
    retries while still inside the window.
  * Elapsed time is measured in UTC, so DST days behave: a time that doesn't exist
    (spring forward, e.g. 02:30) is read with the pre-jump offset and fires at 03:30
    EDT; a repeated hour (fall back) can't fire twice.

Two leftovers from Kael's review of her draft are settled here: occurrences from
before a check-in existed (or from while it was stopped) are never fired or reported,
and a miss more than a day back is still reported, once. The scan looks back up to
LOOKBACK_DAYS, newest first, and stops at the first occurrence that is handled, fires,
or is reported missed.

Delivery is at-least-once (Codex review, 2026-10-12). An occurrence that was sent but
couldn't be recorded is remembered in the watcher until the record lands, and never
sent again by it. A tap CAN still repeat: after a crash between the send and the
record, or when the line accepted a tap but its answer was lost (a timeout reads as not
delivered, and the next tick sends again rather than risk losing it). So each tap names
its scheduled day, and the RECEIVER dedupes on (name, day): Kael's run-book starts a job
at most once per day.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable, Protocol
from zoneinfo import ZoneInfo

import httpx

log = logging.getLogger(__name__)

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
LATE_WINDOW = timedelta(hours=2)
LOOKBACK_DAYS = 7
TICK_S = 60
EASTERN = ZoneInfo("America/New_York")  # the owner's clock, as everywhere in her
TAP_PREFIX = "(check-in)"
# Rows are validated by the table itself (days, HH:MM, non-empty name and message:
# db/migrations/014's CHECK constraints), the one place they are written.


@dataclass
class Checkin:
    name: str
    days: list[str]
    at: str
    message: str
    active: bool = True
    last_fired: date | None = None
    last_skipped: date | None = None
    created_at: datetime | None = None
    resumed_at: datetime | None = None   # the last time it was turned back on

    def not_before(self) -> datetime | None:
        """Occurrences scheduled before this never existed for it (created, or resumed)."""
        marks = [m.astimezone(timezone.utc) for m in (self.created_at, self.resumed_at) if m is not None]
        return max(marks) if marks else None


class CheckinStore(Protocol):
    def active(self) -> list[Checkin]: ...
    def mark_fired(self, name: str, day: date) -> None: ...
    def mark_skipped(self, name: str, day: date) -> None: ...
    def list(self) -> list[Checkin]: ...
    def set_active(self, name: str, active: bool) -> bool: ...


def _scheduled(day: date, hhmm: str) -> datetime:
    # fold=0: a repeated hour means its first pass; a missing one takes the pre-jump offset.
    return datetime(day.year, day.month, day.day, int(hhmm[:2]), int(hhmm[3:]),
                    tzinfo=EASTERN, fold=0)


def run_once(store: CheckinStore, now: datetime, send: Callable[[Checkin, date], bool], *,
             missed: Callable[[Checkin, date], None] | None = None,
             delivered: set | None = None) -> dict:
    """One tick. `now` must be timezone-aware. `send(checkin, day)` is injected; `missed(checkin,
    day)`, when given, is told once about each skipped day (best effort, never retried).
    `delivered` (the watcher's own set of (name, day)) keeps an occurrence that was sent but
    couldn't be recorded from being sent again.

    Returns {"fired": [names], "late": [names fired late],
             "skipped": [{"name", "day", "reason"}], "failed": [{"name", "error"}]}.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    result: dict = {"fired": [], "late": [], "skipped": [], "failed": []}
    now_utc = now.astimezone(timezone.utc)
    today = now.astimezone(EASTERN).date()
    delivered = set() if delivered is None else delivered
    for item in store.active():
        boundary = item.not_before()
        for back in range(LOOKBACK_DAYS + 1):
            day = today - timedelta(days=back)
            if DAYS[day.weekday()] not in item.days:
                continue
            when_utc = _scheduled(day, item.at).astimezone(timezone.utc)   # UTC: DST-safe compares
            if boundary is not None and when_utc < boundary:
                break                       # it didn't exist (or was off) then; nor any earlier day
            if (item.last_fired and item.last_fired >= day) or (item.last_skipped and item.last_skipped >= day):
                break                       # this day (and every earlier one) is handled
            if (item.name, day) in delivered:
                if _record(store, item.name, day):   # sent before; only the record was owed
                    delivered.discard((item.name, day))
                break
            elapsed = now_utc - when_utc
            if elapsed < timedelta(0):
                continue                    # later today: look at the days before it
            if elapsed > LATE_WINDOW:
                store.mark_skipped(item.name, day)
                result["skipped"].append({"name": item.name, "day": day.isoformat(),
                                          "reason": f"missed by more than 2 hours; skipped for {day.isoformat()}"})
                if missed is not None:
                    try:
                        missed(item, day)
                    except Exception:  # best effort: the skip is already recorded and logged
                        log.warning("check-in %s: the missed note failed", item.name, exc_info=True)
                break
            try:
                ok = send(item, day)
            except Exception as exc:  # outside code; any failure means retry next tick
                result["failed"].append({"name": item.name, "error": type(exc).__name__})
                break
            if ok is False:
                result["failed"].append({"name": item.name, "error": "send returned False"})
                break
            delivered.add((item.name, day))   # first: a failed record must not mean a resend
            if _record(store, item.name, day):
                delivered.discard((item.name, day))   # recorded: the set holds only what's owed
            result["fired"].append(item.name)
            if elapsed >= timedelta(minutes=1):
                result["late"].append(item.name)
            break
    return result


def _record(store: CheckinStore, name: str, day: date) -> bool:
    try:
        store.mark_fired(name, day)
        return True
    except Exception:  # the watcher's delivered set holds it; the record is retried next tick
        log.warning("check-in %s: delivered, but recording it failed; will retry the record", name)
        return False


class MemoryCheckinStore:
    """In-process store (tests, dev boxes)."""

    def __init__(self, items: Iterable[Checkin] = ()):
        self.items = {i.name: i for i in items}

    def active(self):
        return [i for i in self.items.values() if i.active]

    def list(self):
        return list(self.items.values())

    def mark_fired(self, name, day):
        self.items[name].last_fired = day

    def mark_skipped(self, name, day):
        self.items[name].last_skipped = day

    def set_active(self, name, active):
        item = self.items.get(name)
        if item is None:
            return False
        if active and not item.active:
            item.resumed_at = datetime.now(timezone.utc)   # what fell due while off never existed
        item.active = active
        return True


class PostgresCheckinStore:
    """v2_kael_checkins (db/migrations/014). Fresh short connections per call, the same
    stance as the storm watcher's seen-store."""

    COLUMNS = "name, days, at_time, message, active, last_fired, last_skipped, created_at, resumed_at"

    def __init__(self, database_url: str):
        self.database_url = database_url

    def _rows(self, where: str = "") -> list[Checkin]:
        import psycopg

        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(f"SELECT {self.COLUMNS} FROM v2_kael_checkins {where} ORDER BY name").fetchall()
        return [Checkin(name=r[0], days=list(r[1]), at=r[2], message=r[3], active=r[4],
                        last_fired=r[5], last_skipped=r[6], created_at=r[7], resumed_at=r[8]) for r in rows]

    def active(self):
        return self._rows("WHERE active")

    def list(self):
        return self._rows()

    def _update(self, sql: str, params: tuple) -> int:
        import psycopg

        with psycopg.connect(self.database_url) as conn:
            return conn.execute(sql, params).rowcount

    def mark_fired(self, name, day):
        self._update("UPDATE v2_kael_checkins SET last_fired = %s, updated_at = now() WHERE name = %s", (day, name))

    def mark_skipped(self, name, day):
        self._update("UPDATE v2_kael_checkins SET last_skipped = %s, updated_at = now() WHERE name = %s", (day, name))

    def set_active(self, name, active):
        # Resuming moves the boundary only when it was off: what fell due meanwhile never existed.
        return self._update(
            "UPDATE v2_kael_checkins SET active = %s, updated_at = now(), "
            "resumed_at = CASE WHEN %s AND NOT active THEN now() ELSE resumed_at END WHERE name = %s",
            (active, active, name)) > 0


def lane_sender(url: str, token: str, client: httpx.Client | None = None) -> Callable[..., bool]:
    """send() for run_once: her line to Kael. 202 = delivered; anything else (the line's
    30 s cooldown, his session offline) is False, so the next tick retries in the window.
    The tap names its scheduled day: the receiver dedupes on (name, day)."""
    http = client or httpx.Client(timeout=10.0)

    def send(item: Checkin, day: date, *, text: str | None = None) -> bool:
        text = text or f"{TAP_PREFIX} {item.name} [{day.isoformat()}]: {' '.join(item.message.split())}"[:2000]
        try:
            resp = http.post(url, json={"message": text, "thread_id": f"checkin:{item.name}:{day.isoformat()}"},
                             headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError:
            log.warning("check-in %s: Kael's line is down", item.name)
            return False
        if resp.status_code != 202:
            log.warning("check-in %s: not delivered (HTTP %s)", item.name, resp.status_code)
            return False
        return True

    return send


class CheckinWatcher:
    def __init__(self, store: CheckinStore, send: Callable[..., bool], *,
                 tick_s: int = TICK_S, clock: Callable[[], datetime] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.store, self.send = store, send
        self._tick_s, self._sleep = tick_s, sleep
        self._clock = clock or (lambda: datetime.now(EASTERN))
        self.delivered: set = set()          # (name, day) sent by this process

    def _missed(self, item: Checkin, day: date) -> None:
        # Labelled MISSED so it can never read as the tap that starts the job.
        self.send(item, day, text=f"{TAP_PREFIX} {item.name} [{day.isoformat()}]: MISSED "
                                  "(more than 2 h late; not retried). Nothing was started.")

    def tick(self) -> dict:
        result = run_once(self.store, self._clock(), self.send, missed=self._missed,
                          delivered=self.delivered)
        for name in result["fired"]:
            log.info("check-in %s: tapped Kael%s", name, " (late)" if name in result["late"] else "")
        for skip in result["skipped"]:
            log.warning("check-in %s: %s", skip["name"], skip["reason"])
        return result

    def run_forever(self) -> None:  # pragma: no cover - thin loop over tick()
        log.info("check-ins up | every %d s", self._tick_s)
        while True:
            try:
                self.tick()
            except Exception:
                log.warning("check-in tick failed", exc_info=True)
            self._sleep(self._tick_s)


def start_checkins(settings) -> threading.Thread | None:
    """Arm-and-forget, --serve only (one watcher, like the storm and presence watchers):
    None unless the database and Kael's line are both configured."""
    if not settings.database_url or not settings.kael_desk_url or settings.kael_desk_token is None:
        return None
    watcher = CheckinWatcher(PostgresCheckinStore(settings.database_url),
                             lane_sender(settings.kael_desk_url, settings.kael_desk_token.get_secret_value()))
    thread = threading.Thread(target=watcher.run_forever, daemon=True, name="kael-checkins")
    thread.start()
    return thread
