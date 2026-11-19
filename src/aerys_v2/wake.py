"""His mornings, from his phone (D74, the owner 2026-11-19; board #177).

The owner's ruling, after a week of grading the rule against his mornings (10 for 10):
  call 1 "i like this": AWAKE (his phone says he woke) = what she held overnight arrives as quiet
         text, nothing spoken; UP (he leaves the bedroom after waking) = her voice comes back and
         the morning brief plays.
  call 2 "B works for me": quiet hours start at 21:00 as before and END when he's up, with 09:00
         (the old fixed end) as the fallback whenever the phone gives nothing.
  naps "your nap bit is good too": daytime sleep gets the nap hold (timely and owed go as quiet
         text, routine waits), and waking from a nap starts no morning routine.

The wake rule is the one graded on #177 (kael-tools wake_check.py): he woke at the first morning
(04:00-13:00 Eastern) update of the phone's sleep-segment sensor, provided a night block came
before it (sleep confidence >= 80 for 3 hours or more since 21:00 the evening before; a reading
that isn't a number breaks the run). A daytime nap never makes one: the rule only looks for the
night's wake. The morning only counts when the settled-room sensor had him IN the bedroom at that
wake; he's UP at the first change after it to somewhere else (unknown/unavailable is no place).
Awake and still in the bedroom stays quiet until he's up, all day if need be, up to the evening's
quiet hours (Codex 11/19 on the first cut's 13:00 cap: "nothing spoken until he's up").

Everything is read from Home Assistant's history, never kept in this process, so a restart
mid-morning reaches the same answer. Unreadable is no evidence, and no evidence is the fixed hours:
missing room history, or a room that reads unknown at the wake or now, counts as no wake (quiet
until 09:00, as before); an unreadable phone means no nap hold. The entity ids come from the
environment (Settings) and are empty by default: unset, none of this runs.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable

from aerys_v2.reach import EASTERN

log = logging.getLogger(__name__)

WINDOW_START_H = 4        # quiet never ends before 04:00, and no wake is looked for earlier
WINDOW_END_H = 13         # the rule's window: no wake is looked for after this
NIGHT_START_H = 21        # the night block counts from here, the evening before
ASLEEP_AT = 80.0          # sleep confidence at or above this is asleep
NIGHT_MIN = 180.0         # minutes of it before a wake counts as the night's
NAP_SETTLE = timedelta(minutes=20)   # a daytime hold needs this long of steady sleep first (not a phone set down)
FRESH_FOR = timedelta(minutes=45)    # a phone reading older than this says nothing
CACHE_S = 60.0
_NOTHING = ("", "unknown", "unavailable", "none")


@dataclass(frozen=True)
class Morning:
    woke_at: datetime | None = None   # the night's wake, today (UTC)
    up_at: datetime | None = None     # the first moment after it outside the bedroom (UTC)


def _num(state) -> float | None:
    try:
        return float(state)
    except (TypeError, ValueError):
        return None


def _local_at(day: date, hour: int) -> datetime:
    return datetime.combine(day, time(hour), tzinfo=EASTERN).astimezone(timezone.utc)


def night_minutes(conf: list, lo: datetime, hi: datetime) -> float:
    """Minutes of sleep confidence >= ASLEEP_AT between lo and hi. conf is HA history (time, state),
    changes only: a value holds until the next one. A reading that isn't a number (unavailable)
    breaks the run: it counts as not asleep, never as the last number carried on (Codex 11/19)."""
    pts = [(t, v if (v := _num(s)) is not None else -1.0) for t, s in conf]
    total = 0.0
    for (t, v), nxt in zip(pts, pts[1:] + [(hi, None)]):
        a, b = max(t, lo), min(nxt[0], hi)
        if v >= ASLEEP_AT and b > a:
            total += (b - a).total_seconds() / 60
    return total


def wake_time(conf: list, seg: list, day: date, now: datetime) -> datetime | None:
    """The night's wake on `day` (the #177 rule), or None: the first sleep-segment update from 04:00
    to 13:00 (and no later than now), if a night block of NIGHT_MIN came before it."""
    lo, hi = _local_at(day, WINDOW_START_H), min(_local_at(day, WINDOW_END_H), now)
    night = _local_at(day - timedelta(days=1), NIGHT_START_H)
    for t, s in seg:
        if lo <= t <= hi and str(s).strip().lower() not in _NOTHING:
            return t if night_minutes(conf, night, t) >= NIGHT_MIN else None   # the first one only
    return None


def _place(state) -> str:
    return str(state).strip().lower()


def room_at(room: list, when: datetime) -> str:
    """The settled room's state at `when` (the last change at or before it), lowercased; '' if none."""
    before = [s for t, s in room if t <= when]
    return _place(before[-1]) if before else ""


def up_time(room: list, woke_at: datetime | None, now: datetime, *, bedroom: str) -> datetime | None:
    """The first CHANGE at or after woke_at (and no later than now) to a real room other than the
    bedroom. Where he was before the wake proves nothing (a stale 'office' from last night): only a
    change after it counts."""
    if woke_at is None:
        return None
    bedroom = _place(bedroom)
    return next((t for t, s in room if woke_at <= t <= now and _place(s) not in (*_NOTHING, bedroom)), None)


def complete_evidence(conf: list, seg: list, room: list, now: datetime) -> bool:
    """All three histories are there, and the room reads a real place at the observation (Codex
    re-check 10: only such a read may settle a no-wake day)."""
    return bool(conf) and bool(seg) and bool(room) and room_at(room, now) not in _NOTHING


def read_morning(conf: list, seg: list, room: list, day: date, now: datetime, *, bedroom: str) -> Morning:
    """Today's morning from the three histories, valid or nothing: the wake counts only if the room
    had him in the bedroom at it; awake-not-up counts only while the room still reads a real place
    (he's in the bedroom). Anything else is no evidence: Morning(), the fixed hours."""
    woke = wake_time(conf, seg, day, now)
    if woke is None or room_at(room, woke) != _place(bedroom):
        return Morning()
    up = up_time(room, woke, now, bedroom=bedroom)
    if up is None and room_at(room, now) in _NOTHING:
        return Morning()                 # the room went dark after the wake: can't tell, fixed hours
    return Morning(woke, up)


def in_fixed_quiet(now: datetime, start_h: int, end_h: int) -> bool:
    """The fixed hours (kael_line.in_quiet_hours' rule): a window that wraps midnight is the usual
    shape; start == end means none."""
    hour = now.astimezone(EASTERN).hour
    if start_h == end_h:
        return False
    if start_h > end_h:
        return hour >= start_h or hour < end_h
    return start_h <= hour < end_h


def _morning_hours(hour: int, start_h: int) -> bool:
    """Where the morning logic applies: from 04:00 until the evening's quiet hours begin."""
    end = start_h if start_h > WINDOW_START_H else 24
    return WINDOW_START_H <= hour < end


def quiet_now(now: datetime, start_h: int, end_h: int, morning: Morning | None) -> bool:
    """D74's quiet hours: from start_h as always; from 04:00 they end when he's up after the night's
    wake. No (valid) wake keeps the fixed end, start_h-end_h; awake but not up keeps them on until
    he's up, at the latest until start_h, when the evening's begin anyway."""
    if start_h == end_h:
        return False
    base = in_fixed_quiet(now, start_h, end_h)
    hour = now.astimezone(EASTERN).hour
    if not _morning_hours(hour, start_h):
        return base
    woke = morning.woke_at if morning else None
    if woke is None or woke > now:
        return base                                  # the phone gave nothing yet: the fixed hours
    up = morning.up_at
    if up is not None and up <= now:
        return False                                 # he's up: quiet's over
    return True                                      # awake in the bedroom: still quiet (text only), no cutoff


class WakeWatch:
    """What the phone says about his mornings and naps. get(path) -> parsed JSON is the HA seam
    (tests); every read is cached CACHE_S, and every failure falls back (logged once a minute)."""

    def __init__(self, get: Callable[[str], object], *, confidence: str, segment: str, room: str,
                 bedroom: str = "bedroom", clock: Callable[[], datetime] | None = None,
                 cache_s: float = CACHE_S, start_h: int = 21):
        self.get, self.confidence, self.segment, self.room = get, confidence, segment, room
        self.bedroom, self.start_h = bedroom, start_h
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.cache_s = cache_s
        self._lock = threading.Lock()
        self._morning: tuple[datetime, date, Morning, bool] | None = None    # (seen at, day, morning, settled)
        self._asleep: tuple[datetime, bool] | None = None

    def _history(self, start: datetime, end: datetime, entities: list[str]) -> dict:
        from urllib.parse import quote

        path = (f"/api/history/period/{quote(start.isoformat())}?end_time={quote(end.isoformat())}"
                f"&filter_entity_id={','.join(entities)}&minimal_response&no_attributes")
        out: dict = {}
        for series in self.get(path) or []:
            if not series:
                continue
            entity = series[0].get("entity_id")
            pts = []
            for p in series:
                raw = p.get("last_changed") or p.get("last_updated")
                try:
                    t = datetime.fromisoformat(str(raw))
                except ValueError:
                    continue
                pts.append((t if t.tzinfo else t.replace(tzinfo=timezone.utc), p.get("state")))
            out[entity] = sorted(pts, key=lambda x: x[0])
        return out

    def morning(self, now: datetime | None = None) -> Morning:
        """Today's wake and up, read from history (None parts when there's nothing, or no reading)."""
        now = now or self.clock()
        local = now.astimezone(EASTERN)
        day = local.date()
        if not _morning_hours(local.hour, self.start_h):
            return Morning()                         # nothing to look for outside the morning
        with self._lock:
            hit = self._morning
            if hit and hit[1] == day and (hit[3] or (now - hit[0]).total_seconds() < self.cache_s):
                return hit[2]
        settled = False
        try:
            h = self._history(_local_at(day - timedelta(days=1), NIGHT_START_H) - timedelta(hours=3), now,
                              [self.confidence, self.segment, self.room])
            conf, seg, room = h.get(self.confidence, []), h.get(self.segment, []), h.get(self.room, [])
            m = read_morning(conf, seg, room, day, now, bedroom=self.bedroom)
            # Settled for the day only by what was OBSERVED (Codex re-check 10): an up (it doesn't change),
            # or a no-wake read at or after 13:00 (the rule's window is closed) from complete, valid
            # evidence. Anything else is retried after the minute's cache.
            settled = m.up_at is not None or (m.woke_at is None and local.hour >= WINDOW_END_H
                                              and complete_evidence(conf, seg, room, now))
        except Exception as e:
            log.warning("wake: couldn't read his morning (%s); the fixed hours stand", type(e).__name__)
            m = Morning()
        with self._lock:
            self._morning = (now, day, m, settled)
        return m

    def quiet(self, start_h: int, end_h: int) -> Callable[[datetime], bool]:
        """quiet(now) for the desk: D74's hours (quiet_now)."""
        def quiet(now: datetime) -> bool:
            return quiet_now(now, start_h, end_h, self.morning(now))
        return quiet

    def awake(self, now: datetime | None = None) -> bool:
        """He woke this morning and isn't up yet: what she holds goes as quiet text, nothing aloud."""
        now = now or self.clock()
        m = self.morning(now)
        return m.woke_at is not None and m.woke_at <= now and (m.up_at is None or m.up_at > now)

    def asleep(self, now: datetime | None = None) -> bool:
        """The phone says he's asleep right now, and has for NAP_SETTLE: a nap (or a sleep-in past the
        fixed hours). A stale or unreadable phone says nothing: no hold."""
        now = now or self.clock()
        with self._lock:
            hit = self._asleep
            if hit and (now - hit[0]).total_seconds() < self.cache_s:
                return hit[1]
        try:
            cur = self.get(f"/api/states/{self.confidence}") or {}
            seen_raw = cur.get("last_reported") or cur.get("last_updated") or cur.get("last_changed")
            seen = datetime.fromisoformat(str(seen_raw))
            seen = seen if seen.tzinfo else seen.replace(tzinfo=timezone.utc)
            value = _num(cur.get("state"))
            yes = value is not None and value >= ASLEEP_AT and now - seen <= FRESH_FOR
            if yes:      # steady sleep, not a phone just set down: every reading in NAP_SETTLE was asleep
                recent = self._history(now - NAP_SETTLE, now, [self.confidence]).get(self.confidence, [])
                nums = [v for t, s in recent if (v := _num(s)) is not None]
                changed = datetime.fromisoformat(str(cur.get("last_changed") or seen_raw))
                changed = changed if changed.tzinfo else changed.replace(tzinfo=timezone.utc)
                yes = bool(nums) and min(nums) >= ASLEEP_AT and (len(nums) > 1 or now - changed >= NAP_SETTLE)
        except Exception as e:
            log.warning("wake: couldn't read his phone (%s); no nap hold", type(e).__name__)
            yes = False
        with self._lock:
            self._asleep = (now, yes)
        return yes


def wake_watch_for(settings, *, get: Callable[[str], object] | None = None,
                   clock: Callable[[], datetime] | None = None) -> WakeWatch | None:
    """None unless armed: both phone sensors named (WAKE_SLEEP_CONFIDENCE_ENTITY and
    WAKE_SLEEP_SEGMENT_ENTITY), a settled-room sensor, and HA."""
    confidence = str(getattr(settings, "wake_sleep_confidence_entity", "") or "").strip()
    segment = str(getattr(settings, "wake_sleep_segment_entity", "") or "").strip()
    room = str(getattr(settings, "presence_settled_entity", "") or "").strip()
    if not (confidence and segment and room) or getattr(settings, "ha_token", None) is None:
        return None
    if get is None:
        import httpx

        base = str(getattr(settings, "ha_base_url", "")).rstrip("/")
        headers = {"Authorization": f"Bearer {settings.ha_token.get_secret_value()}"}

        def get(path: str):
            r = httpx.get(f"{base}{path}", headers=headers, timeout=10.0)
            r.raise_for_status()
            return r.json()

    return WakeWatch(get, confidence=confidence, segment=segment, room=room,
                     bedroom=str(getattr(settings, "wake_bedroom_room", "bedroom") or "bedroom"), clock=clock,
                     start_h=int(getattr(settings, "owner_quiet_start_hour", 21)))
