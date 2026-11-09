"""Her welcome-home hello (board #178; #176 Stage 2, "present").

The owner, 2026-11-09, after Kael greeted him unasked when he got home: "this is what i like.
getting a welcome home message without me having to fish for one", and that's what his September
ask (#19) was for. His calls: "the 30+ minutes away works for me"; and "we can be adaptive with
this, she could be inquisitive if i left without anything on the schedule or she could ask how my
appointment went". With the design he liked ("yeah i like this"): if the morning brief is still
due, the hello carries it (one message, and it knows he's been up and out); if a follow-up was
going to ask about that appointment, the hello asks instead (never asked twice).

So, once per arrival:
  1. his settled home/away sensor turned home, and HA's own history says the stretch before it
     was away (not a sensor gap or an HA restart) for away_minutes or more. The history is the
     durable record: a brain restart while he's settling loses nothing;
  2. the house settles him in a room with a speaker (never the bedroom unless configured) within
     settle_minutes of getting home; if it doesn't, nothing is said (a dropped row says why);
  3. not in quiet hours (a welcome home the next morning is wrong: it's dropped), and nothing
     holds her voice (a meeting or a nap: it waits, up to HELLO_FOR after he got home). The one
     exception, his D18 #3 (11/09, "ok"): home inside quiet hours with the morning brief still due,
     it's said in the brief's room, carrying the brief, the way the brief itself speaks there
     before 9 (both are brief rows at the desk). Coming home counts as being up. Settled anywhere
     else, it's dropped and the brief goes on its own when he reaches its room.
She composes it from facts only: how long he was out; what was on his personal calendars while
he was out (she asks how it went unless a check-in already asked, he muted appointments, or his
check-in dial is used up today); what was held for him; and the morning brief's facts when that's
still due. It goes to the desk pinned to that room and outside the budget, like the briefs;
company mode turns it into a notification. If her words can't be had, a plain version goes.

What she says, and what his calendar says, is the private part: never logged.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from aerys_v2.brief import Facts, _at, _csv, _ha, _loop, held_detail, plain_brief
from aerys_v2.followups import (CHECK_IN_TOPICS, CHECKIN_FOR, CHECKIN_LOCK, DATA_ONLY, HELLO_KEPT, Appointment,
                                appointment_key)
from aerys_v2.followups import SOURCE as FOLLOWUP_SOURCE
from aerys_v2.followups import TOPIC as FOLLOWUP_TOPIC
from aerys_v2.reach import EASTERN, ReachEvent, local_day_start, muted_topics, next_local

log = logging.getLogger(__name__)

SOURCE = "welcome"
TOPIC = "welcome"           # he can mute it in words (tools/reach_prefs.py)
HELLO_FOR = timedelta(hours=1)        # past this after he got home, it's no longer a welcome
LOOKBACK = timedelta(hours=24)        # how far back HA's history is read for when he left
GAP_STATES = {"unknown", "unavailable", ""}   # a sensor gap or an HA restart: not a trip
TITLE_MAX = 120
KEPT = HELLO_KEPT                            # a held check-in's reason: f"{KEPT} ({arrival key})"
ASKED = "asked in the welcome-home hello"     # its reason once a hello asked it
RECONCILE_EVERY = timedelta(minutes=5)
ASK, NO_ASK = "ask", "don't ask"                 # how the facts mark what she may ask about

WELCOME_PROMPT = (
    "\n\n[Welcome home] Chris just got home after being out for a while, and the house has him "
    "settled in a room with you. Say hi, out loud: one to three short sentences, warm, first "
    "person, your own voice. If the facts list something from his calendar marked [ask], ask "
    "lightly how it went: one question. Never ask about anything marked [don't ask]: you may know "
    "he went, but you asked already, he'd rather you didn't, or it can wait. If nothing was on his calendar while he was out, you may "
    "ask one light, curious question about what he got up to; never pry, never guess where he "
    "was, and it's fine if he doesn't say. If things came in while he was out, say so in a few "
    "words; the details are in his messages. If his morning brief is included, give it after the "
    "hello in two to four more short sentences: the weather, what's left on today's calendar "
    "(work meetings: how many and when, never what they're about), anything that came in "
    "overnight. He's been up and out, so never talk as if he just woke up. Use ONLY the facts "
    "below and never add any. Say times the way people say them (\"ten thirty\"). Nothing to read "
    "off a screen: no lists, no markdown, no emoji. Reply with ONLY what to say."
)


@dataclass
class HomeReading:
    home: str = ""                    # his settled home/away state, lowercased
    since: datetime | None = None     # when it last changed: when he got home, if he's home
    room: str = ""                    # the settled room, lowercased
    confirmed: bool = False           # motion in that room backs his phone up


@dataclass
class Visit:
    title: str
    when: str                         # "10:00 AM to 11:30 AM"
    ask: bool                         # the hello may ask how it went
    claim: int | None = None          # the follow-up row the hello holds for it (folded after)


@dataclass
class WelcomeFacts:
    away: timedelta
    room: str = ""
    visits: list = field(default_factory=list)   # Visit: his personal calendar while he was out
    held: list = field(default_factory=list)     # Reach rows held for him while he was out
    brief: Facts | None = None                   # the morning brief's facts, when it comes along


def _utc(dt: datetime) -> datetime:
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _clock(dt: datetime) -> str:
    """'1:30 PM', Eastern (the calendar tool's format, without importing the tools)."""
    return dt.astimezone(EASTERN).strftime("%I:%M %p").lstrip("0")


def how_long(away: timedelta) -> str:
    """'about 45 minutes', 'about an hour and a half', 'about 3 hours', 'a day or more'."""
    if away >= LOOKBACK:
        return "a day or more"
    minutes = int(away.total_seconds() // 60)
    if minutes < 50:
        return f"about {minutes} minutes"
    hours, half = divmod(round(minutes / 30), 2)
    if hours == 1:
        return "about an hour" + (" and a half" if half else "")
    return f"about {hours}" + (" and a half" if half else "") + " hours"


def left_at(history: list, arrived: datetime, window_start: datetime) -> datetime | None:
    """When he left, from the home sensor's history before `arrived`: the start of the run of
    away states that ends at his arrival. None when that run has no real away state in it (a
    sensor gap or an HA restart while he was home is not a trip). A run older than the window
    started at window_start or before."""
    states = []
    for s in history or []:
        if not isinstance(s, dict):
            continue
        try:
            t = _utc(datetime.fromisoformat(str(s.get("last_changed")).replace("Z", "+00:00")))
        except ValueError:
            continue
        if t < arrived:
            states.append((t, str(s.get("state") or "").strip().lower()))
    away, start = False, None
    for t, state in sorted(states, reverse=True):
        if state == "home":
            return start if away else None
        if state not in GAP_STATES:
            away = True
        start = t
    return min(start, window_start) if away and start is not None else None


def welcome_facts_text(facts: WelcomeFacts) -> str:
    """The facts as the model reads them, framed as data: a calendar title can come from anyone
    who sends him an invite."""
    lines = [f"He was out for {how_long(facts.away)}; he's in the {facts.room or 'house'} now."]
    if facts.visits:
        lines.append("On his calendar while he was out:")
        lines += [f"- [{ASK if v.ask else NO_ASK}] {v.when}: {v.title}" for v in facts.visits]
    else:
        lines.append("On his calendar while he was out: nothing")
    if facts.held:
        lines.append("Came in while he was out (the details are in his messages):")
        lines += [f"- {r.headline}" for r in facts.held]
    b = facts.brief
    if b is not None:
        lines.append("His morning brief (still due today):")
        lines.append(f"- Weather today: {b.weather or 'unknown (say nothing about it)'}")
        lines.append("- Left on today's calendar: " + ("; ".join(b.events) if b.events else "nothing"))
        overnight = [*b.held, *b.quiet]
        lines.append("- Came in overnight: " + ("; ".join(r.headline for r in overnight) if overnight else "nothing"))
    return DATA_ONLY + "\n".join(lines)


def plain_welcome(facts: WelcomeFacts) -> str:
    """The hello without her words (the model failed or answered nothing): plainly."""
    parts = ["Welcome home."]
    ask = [v for v in facts.visits if v.ask]
    if ask:
        parts.append(f"How did {ask[0].title} go?")
    elif not facts.visits:
        parts.append("Get up to anything good while you were out?")
    if len(facts.held) == 1:
        parts.append("One thing came in while you were out; it's in your messages.")
    elif facts.held:
        parts.append(f"{len(facts.held)} things came in while you were out; they're in your messages.")
    if facts.brief is not None:
        parts.append(plain_brief(facts.brief).removeprefix("Good morning.").strip())
    return " ".join(p for p in parts if p)


class WelcomeHome:
    """tick() -> what happened this pass (logged, never the words). Every leg is injected:
    read() -> HomeReading; away_since(arrived) -> when he left, or None (not a trip);
    visits(left, arrived) -> the Appointments on his personal calendars in between;
    compose(WelcomeFacts) -> her words. brief: the MorningBrief, when armed (its lock is shared,
    so the two never compose at once). followups: the follow-up store, when armed."""

    def __init__(self, desk, *, read: Callable[[], HomeReading],
                 away_since: Callable[[datetime], datetime | None],
                 compose: Callable[[WelcomeFacts], str],
                 visits: Callable[[datetime, datetime], list] | None = None,
                 rooms: tuple = ("office", "kitchen"), brief=None, followups=None,
                 away_minutes: int = 30, settle_minutes: int = 20, quiet_start_hour: int = 21,
                 clock: Callable[[], datetime] | None = None):
        self.desk, self.read, self.away_since, self.compose, self.visits = desk, read, away_since, compose, visits
        self.rooms = tuple(r.strip().lower() for r in rooms if r.strip())
        self.brief, self.followups = brief, followups
        self.away_for = timedelta(minutes=away_minutes)
        self.settle_for = timedelta(minutes=settle_minutes)
        self.quiet_start_hour = quiet_start_hour
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.lock = brief.lock if brief is not None else threading.Lock()
        self.left: dict = {}          # arrival key -> when he left (None: not a trip)
        self.arrived: dict = {}       # arrival key -> when he got home
        self.finished: list = []      # arrival keys this process is done with (the last few)
        self.last_reconcile: datetime | None = None
        if brief is not None:
            brief.defer = self.greeting   # the brief waits while an arrival is still being greeted

    def _finish(self, key: str) -> None:
        self.finished = [*self.finished[-19:], key]
        self.left.pop(key, None)
        self.arrived.pop(key, None)

    def greeting(self, now: datetime) -> bool:
        """True while the hello waits for him to settle after a trip (not finished, inside
        settle_for): the morning brief waits, so the hello can carry it rather than race it to his
        room. Only that long, so a brief can't be held past its cutoff. Called from the brief's
        thread: if the arrivals change under it, it says wait (the next pass asks again)."""
        try:
            trips = [(key, left, self.arrived.get(key)) for key, left in list(self.left.items())]
        except RuntimeError:
            return True
        for key, left, arrived in trips:
            if left is None or arrived is None or key in self.finished:
                continue
            if arrived - left >= self.away_for and now - arrived <= self.settle_for:
                return True
        return False

    def _drop(self, key: str, now: datetime, reason: str) -> None:
        """Recorded, so a restart doesn't take it up again: 'why didn't she say hi?' has an answer."""
        try:
            self.desk.store.add(ReachEvent(source=SOURCE, topic=TOPIC, headline="(a welcome home, not said)",
                                           brief=True, dedupe_key=key), now, status="dropped", reason=reason,
                                not_before=None)
        except Exception:
            log.warning("welcome home: couldn't record why it wasn't said; this process remembers", exc_info=True)

    def tick(self) -> str:
        now = self.clock()
        self._reconcile(now)
        try:
            r = self.read()
        except Exception:
            log.warning("welcome home: can't read the house this pass", exc_info=True)
            return "no reading"
        if r.home != "home" or r.since is None:
            return "not home"
        arrived = _utc(r.since)
        key = f"{SOURCE}:{arrived.isoformat(timespec='seconds')}"
        if key in self.finished:
            return "done"
        if now - arrived > HELLO_FOR:
            if self.left.get(key) is not None:          # a trip this process was waiting to greet
                self._drop(key, now, "never got to say it")
            self._finish(key)
            return "too late"
        store = self.desk.store
        if store.decided(key, arrived - timedelta(minutes=1)):     # said, or decided, before a restart
            self._finish(key)
            return "done"
        if key not in self.left:
            try:
                self.left[key] = self.away_since(arrived)
                self.arrived[key] = arrived
            except Exception:
                log.warning("welcome home: couldn't read when he left; tried again next pass", exc_info=True)
                return "no history"
        left = self.left[key]
        if left is None:
            self._finish(key)
            return "not a trip"
        if arrived - left < self.away_for:
            self._finish(key)
            return "a short trip"
        if TOPIC in muted_topics(store.prefs(self.desk.budget_default)):
            self._drop(key, now, f"you muted {TOPIC}")
            self._finish(key)
            return "muted"
        # Inside quiet hours only a hello that carries the morning brief is said, in the brief's room.
        night = self.desk.quiet(now)
        if night and not (self.brief is not None and self._brief_due(now, now.astimezone(EASTERN).date())):
            self._drop(key, now, "quiet hours")
            self._finish(key)
            return "quiet hours"
        if r.room not in self.rooms or not r.confirmed:
            if now - arrived > self.settle_for:
                self._drop(key, now, "never settled in a room with a speaker")
                self._finish(key)
                return "never settled"
            return "waiting to settle"
        if night and r.room != self.brief.room:
            self._drop(key, now, "quiet hours")
            self._finish(key)
            return "quiet hours"
        held_by = self.desk.voice_held(now)
        if held_by:
            return f"held: {held_by}"
        with self.lock:
            return self._say(key, arrived, left, r.room, now, must_carry=night)

    def _brief_due(self, now: datetime, day: date) -> bool:
        b = self.brief
        if b is None or b.done == day:
            return False
        if not (b.start_hour <= now.astimezone(EASTERN).hour < b.cutoff_hour):
            return False
        key, start = f"brief:{day.isoformat()}", _at(day, b.start_hour)
        store = self.desk.store
        if store.recently_sent(key, start) or store.carried(key, start) is not None:
            return False
        return "brief" not in muted_topics(store.prefs(self.desk.budget_default))

    def _claim(self, a: Appointment, now: datetime, kept: str) -> int | None:
        """Hold an appointment's check-in for this hello, so it's asked once. Its follow-up row
        becomes 'skipped' with `kept` (KEPT and the arrival) until the hello is said (then
        'raised', in _reconcile) or doesn't go (then it goes as a check-in after all: _release).
        A row the follow-ups made is taken only once it's settled ('raised'), and only if its
        check-in hasn't been said: it's marked ours FIRST, so a restart from here is reconciled,
        then its check-in is taken back from the desk. A restart that finds the row already ours
        (this same arrival) picks it up again. None when it isn't the hello's to ask."""
        fs = self.followups
        had = fs.by_key(a.key)
        if had is None:
            made = fs.add("appointment", a.title, now, now, status="skipped", source_key=a.key, reason=kept,
                          expires_at=now + CHECKIN_FOR)
            if made is not None:
                return made.id
            had = fs.by_key(a.key)               # the follow-ups got there first
            if had is None:
                return None
        ours = had.status == "skipped" and had.reason == kept
        if not ours and (had.status != "raised" or had.reason == ASKED):
            return None      # in flight, not worth asking, the dial was used, or an earlier hello asked it
        if not ours:
            fs.mark(had.id, now, status="skipped", reason=kept)
        if self.desk.take_back(f"followup:{had.id}", now, reason=ASKED) == "went":
            fs.mark(had.id, now, status="raised", reason="desk: it went on its own")   # never asked twice
            return None
        return had.id

    def _visits(self, left: datetime, arrived: datetime, now: datetime, key: str) -> list:
        """His personal calendar while he was out. She may ask about ONE of them (the one that ended
        last), if appointments aren't muted, his check-in dial has room today, and no check-in
        already asked; the rest she knows about but leaves to the follow-ups. A claim this arrival
        already held (a restart while she composed) that she doesn't ask about now goes as a
        check-in instead (_release), so nothing is marked asked that wasn't."""
        if self.visits is None:
            return []
        try:
            found = self.visits(left, arrived)
        except Exception:
            log.warning("welcome home: the calendars couldn't be read; going without them", exc_info=True)
            return []
        with CHECKIN_LOCK:     # his dial: counted and reserved in one step with the follow-ups runner's
            chosen, mine = self._choose(found, now, key)
        for f in mine:
            if f.id not in chosen.values():
                try:
                    self._release(f, now, "not in the welcome-home hello after all")
                except Exception:
                    log.warning("welcome home: follow-up %s: releasing it failed; reconciled later", f.id,
                                exc_info=True)
        return [Visit(title=" ".join(a.title.split())[:TITLE_MAX] or "something",
                      when=f"{_clock(a.start)} to {_clock(a.end)}", ask=a.key in chosen, claim=chosen.get(a.key))
                for a in sorted(found, key=lambda a: a.start)]

    def _choose(self, found: list, now: datetime, key: str) -> tuple[dict, list]:
        """{appointment key: its claim} for the one she'll ask about (if any), and the claims this
        arrival already held. Called under CHECKIN_LOCK."""
        kept = f"{KEPT} ({key})"
        mine = []
        prefs = self.desk.store.prefs(self.desk.budget_default)
        may_ask = "appointments" not in muted_topics(prefs)
        if self.followups is not None:
            try:
                mine = [f for f in self.followups.kept_for(kept) if f.reason == kept]
                if may_ask:     # what's used today, not counting what this hello already holds
                    used = self.followups.raised_since(local_day_start(now)) - len(mine)
                    may_ask = used < prefs.checkins
            except Exception:
                log.warning("welcome home: the check-in count couldn't be read; not asking", exc_info=True)
                may_ask = False
        else:
            may_ask = may_ask and prefs.checkins > 0
        chosen: dict = {}
        for a in sorted(found, key=lambda a: a.end, reverse=True):
            if not may_ask:
                break
            if self.followups is None:
                chosen[a.key] = None
                break
            try:
                claim = self._claim(a, now, kept)
            except Exception:
                log.warning("welcome home: couldn't hold a check-in for the hello; not asking", exc_info=True)
                break
            if claim is not None:
                chosen[a.key] = claim
                break
        return chosen, mine

    def _release(self, f, now: datetime, why: str) -> None:
        """The hello that held this check-in didn't go: it goes as an ordinary check-in instead, by
        the follow-ups' own rules (his mute, his dial, the expiry fixed when it was made)."""
        fs = self.followups
        prefs = self.desk.store.prefs(self.desk.budget_default)
        if f.expires_at is not None and now >= f.expires_at:
            fs.mark(f.id, now, status="skipped", reason=f"{why}; too late for a check-in")
            return
        if "appointments" in muted_topics(prefs):
            fs.mark(f.id, now, status="skipped", reason=f"{why}; appointments muted")
            return
        held = 1 if f.status == "skipped" and f.reason.startswith(KEPT) else 0    # it counts itself
        if fs.raised_since(local_day_start(now)) - held >= prefs.checkins:
            fs.mark(f.id, now, status="skipped", reason=f"{why}; today's check-ins were used")
            return
        text = f.message or f"How did {f.about} go?"
        outcome = self.desk.reach_out(ReachEvent(
            source=FOLLOWUP_SOURCE["appointment"], topic=FOLLOWUP_TOPIC["appointment"], headline=text,
            dedupe_key=f"followup:{f.id}", expires_at=f.expires_at))
        went = outcome in ("sent", "queued", "duplicate")
        fs.mark(f.id, now, status="raised" if went else "skipped", reason=f"{why}; as a check-in: {outcome}",
                message=text)

    def _reconcile(self, now: datetime) -> None:
        """Claims whose hello has settled: said -> 'raised'; dropped, failed, or never written (a
        restart while she composed) -> a check-in after all. Still queued, or possibly still on
        its way: left alone."""
        if self.followups is None or (self.last_reconcile and now - self.last_reconcile < RECONCILE_EVERY):
            return
        self.last_reconcile = now
        try:
            for f in self.followups.kept_for(KEPT):
                key = f.reason[len(KEPT):].strip().strip("()")
                states = self.desk.store.statuses(key)
                if "sent" in states:
                    self.followups.mark(f.id, now, status="raised", reason=ASKED)
                    continue
                if "queued" in states:
                    continue
                if not states:
                    try:
                        arrived = _utc(datetime.fromisoformat(key.split(":", 1)[1]))
                    except (IndexError, ValueError):
                        arrived = None
                    if arrived is not None and now - arrived <= HELLO_FOR + RECONCILE_EVERY:
                        continue          # it may still be on its way
                self._release(f, now, "the welcome-home hello didn't go")
        except Exception:
            log.warning("welcome home: reconciling held check-ins failed; tried again later", exc_info=True)

    def _say(self, key: str, arrived: datetime, left: datetime, room: str, now: datetime, *,
             must_carry: bool = False) -> str:
        day = now.astimezone(EASTERN).date()
        # Never said at night, and not long after he got home (the desk drops it past this).
        expires = min(arrived + HELLO_FOR, next_local(arrived, self.quiet_start_hour))
        store = self.desk.store
        brief_facts, marker = None, None
        if self._brief_due(now, day):
            # The marker goes first, and the hello carries it in its folds: the brief counts as given
            # only while that hello is queued or said (store.carried), so a hello that never goes, or
            # a restart before it's written, leaves the brief free to go on its own.
            try:
                marker = store.add(ReachEvent(source="brief", topic="brief", headline="(carried by the welcome-home hello)",
                                              brief=True, dedupe_key=f"brief:{day.isoformat()}"), now, status="folded",
                                   reason="carried by the welcome-home hello", not_before=None)
            except Exception:
                log.warning("welcome home: couldn't mark the brief as carried; it goes on its own", exc_info=True)
        if must_carry and marker is None:
            # Before 9 it's said only as the brief's carrier (D18 #3); the brief went meanwhile, or
            # couldn't be marked: a plain hello stays out of quiet hours.
            self._drop(key, now, "quiet hours")
            self._finish(key)
            return "quiet hours"
        if marker is not None:
            since, cutoff = _at(day - timedelta(days=1), 18), _at(day, self.brief.cutoff_hour)
            expires = min(expires, cutoff)
            # Reserved first, as the brief does: what it carries can't also go out on its own.
            held = self.desk.reserve_for_brief(since, cutoff, reason="carried by the welcome-home hello",
                                               skip_topics=CHECK_IN_TOPICS)
            quiet = [q for q in store.sent_quietly(since) if not q.text_only]
            try:
                brief_facts = self.brief.gather(now, held, quiet)
            except Exception:
                log.warning("welcome home: the brief's day couldn't be gathered; going with what she holds",
                            exc_info=True)
                brief_facts = Facts(held=held, quiet=quiet)
        else:
            held = self.desk.reserve_for_brief(left, expires, reason="carried by the welcome-home hello",
                                               skip_topics=CHECK_IN_TOPICS)
        visits = self._visits(left, arrived, now, key)
        facts = WelcomeFacts(away=arrived - left, room=room, visits=visits, held=[] if marker else held,
                             brief=brief_facts)
        text = ""
        try:
            text = (self.compose(facts) or "").strip()
        except Exception:
            log.warning("welcome home: her words failed; the plain version goes", exc_info=True)
        if not text:
            text = plain_welcome(facts)
        folds = tuple(h.id for h in held) + ((marker.id,) if marker is not None else ())
        outcome = self.desk.reach_out(ReachEvent(
            source=SOURCE, topic=TOPIC, headline=text, detail=held_detail(held), room=room, brief=True,
            folds=folds, dedupe_key=key, expires_at=expires))
        for v in visits:
            if v.claim is None or outcome in ("queued", "duplicate"):   # queued: settled by _reconcile
                continue
            try:
                if outcome == "sent":
                    self.followups.mark(v.claim, now, status="raised", reason=ASKED)
                else:
                    self._release(self.followups.get(v.claim), now, f"the welcome-home hello was {outcome}")
            except Exception:
                log.warning("welcome home: follow-up %s: settling it failed; reconciled later", v.claim, exc_info=True)
        self._finish(key)
        log.info("welcome home: %s (%d held, %d on the calendar, brief %s)", outcome, len(held), len(visits),
                 "carried" if marker is not None else "not carried")
        return outcome

    def run_forever(self) -> None:  # pragma: no cover - thin loop over tick()
        log.info("welcome home armed | %s, after %d min away", "/".join(self.rooms),
                 int(self.away_for.total_seconds() // 60))
        _loop(self.tick, "welcome home")


def start_welcome_home(hello: "WelcomeHome | None") -> threading.Thread | None:
    if hello is None:
        return None
    thread = threading.Thread(target=hello.run_forever, daemon=True, name="welcome-home")
    thread.start()
    return thread


# --- production legs (wired from settings by welcome_home_for) ----------------------------


def home_reader_for(settings, *, post=None) -> Callable[[], HomeReading]:
    """One HA template call: his settled home/away state and when it last changed, and the
    settled room with its confidence. post(url, headers, json) is the test seam."""
    import httpx

    home = str(getattr(settings, "welcome_home_entity", "") or "").strip()
    settled = str(getattr(settings, "presence_settled_entity", "") or "").strip()
    template = ("{{ {'home': states(" + repr(home) + "), 'since': (expand(" + repr(home)
                + ") | map(attribute='last_changed') | map('string') | list), 'room': states(" + repr(settled)
                + "), 'confidence': state_attr(" + repr(settled) + ", 'confidence')} | to_json }}")
    base, headers = _ha(settings)
    send = post or (lambda url, hdrs, body: httpx.post(url, headers=hdrs, json=body, timeout=5.0))

    def read() -> HomeReading:
        r = send(f"{base}/api/template", headers, {"template": template})
        r.raise_for_status()
        data = json.loads(r.text)
        since = None
        for raw in data.get("since") or []:
            try:
                since = _utc(datetime.fromisoformat(str(raw)))
            except ValueError:
                continue
        return HomeReading(home=str(data.get("home") or "").strip().lower(), since=since,
                           room=str(data.get("room") or "").strip().lower(),
                           confirmed=str(data.get("confidence") or "") == "confirmed")

    return read


def away_since_for(settings, *, get=None) -> Callable[[datetime], datetime | None]:
    """When he left, from HA's recorder (the home sensor's history over the LOOKBACK before he
    got home). get(url, headers) is the test seam."""
    import httpx
    from urllib.parse import quote, urlencode

    entity = str(getattr(settings, "welcome_home_entity", "") or "").strip()
    base, headers = _ha(settings)
    fetch = get or (lambda url, hdrs: httpx.get(url, headers=hdrs, timeout=10.0))

    def away_since(arrived: datetime) -> datetime | None:
        start = arrived - LOOKBACK
        url = (f"{base}/api/history/period/{quote(start.isoformat())}?"
               + urlencode({"filter_entity_id": entity, "end_time": arrived.isoformat()})
               + "&minimal_response&no_attributes")
        r = fetch(url, headers)
        r.raise_for_status()
        data = r.json()
        rows = data[0] if isinstance(data, list) and data and isinstance(data[0], list) else []
        return left_at(rows, arrived, start)

    return away_since


def visits_for(settings, *, get=None) -> Callable[[datetime, datetime], list]:
    """Timed events on his PERSONAL calendars (brief_calendars, never owner_meeting_calendars)
    that overlap the time he was out, keyed as the follow-ups key them. A calendar that can't be
    read is skipped (logged)."""
    import httpx
    from urllib.parse import urlencode

    from aerys_v2.tools.calendar_events import _parse_when

    work = set(_csv(getattr(settings, "owner_meeting_calendars", "")))
    personal = [e for e in _csv(getattr(settings, "brief_calendars", "")) if e not in work]
    base, headers = _ha(settings)
    fetch = get or (lambda url, hdrs: httpx.get(url, headers=hdrs, timeout=5.0))

    def read(left: datetime, arrived: datetime) -> list:
        window = urlencode({"start": left.isoformat(), "end": arrived.isoformat()})
        out = []
        for entity in personal:
            try:
                r = fetch(f"{base}/api/calendars/{entity}?{window}", headers)
                r.raise_for_status()
                items = r.json()
                if not isinstance(items, list):
                    raise ValueError("not a list of events")
            except Exception as e:
                log.warning("welcome home: a calendar read failed (%s)", type(e).__name__)
                continue
            for ev in items:
                if not isinstance(ev, dict) or str(ev.get("status") or "").upper() == "CANCELLED":
                    continue
                start, _ = _parse_when(ev.get("start"))
                end, _ = _parse_when(ev.get("end"))
                if start is None or end is None:
                    continue                      # all-day: not somewhere he went
                start = start if start.tzinfo else start.replace(tzinfo=EASTERN)
                end = end if end.tzinfo else end.replace(tzinfo=EASTERN)
                if end <= left or start >= arrived:
                    continue
                out.append(Appointment(key=appointment_key(entity, ev, start),
                                       title=str(ev.get("summary") or "").strip() or "something",
                                       start=start, end=end, location=str(ev.get("location") or "").strip(),
                                       repeats=bool(ev.get("rrule") or ev.get("recurrence_id"))))
        return sorted(out, key=lambda a: a.start)

    return read


def welcome_home_for(settings, desk, *, brief=None) -> WelcomeHome | None:
    """None unless armed: welcome_home on, its entity, a desk, HA, and a speaker in one of its rooms."""
    entity = str(getattr(settings, "welcome_home_entity", "") or "").strip()
    if (not getattr(settings, "welcome_home", False) or not entity or desk is None
            or getattr(settings, "ha_token", None) is None):
        return None
    from aerys_v2.brief import composer_for
    from aerys_v2.kael_line import parse_room_devices

    speakers = parse_room_devices(getattr(settings, "reach_room_devices", ""))
    wanted = _csv(getattr(settings, "welcome_home_rooms", ""))
    rooms = tuple(r for r in ([w.lower() for w in wanted] if wanted else [s for s in speakers if s != "bedroom"])
                  if r in speakers)
    if not rooms:
        log.warning("welcome home: no speaker in any of its rooms; not armed")
        return None
    # Its claims (one question per appointment, his daily dial) live in the follow-ups' table
    # (migration 020) whether or not their runner is on.
    followups = None
    if getattr(settings, "database_url", None):
        from aerys_v2.followups import PostgresFollowupStore

        followups = PostgresFollowupStore(settings.database_url)
    return WelcomeHome(
        desk, read=home_reader_for(settings), away_since=away_since_for(settings),
        compose=composer_for(settings, prompt=WELCOME_PROMPT, text_of=welcome_facts_text),
        visits=visits_for(settings), rooms=rooms, brief=brief, followups=followups,
        away_minutes=int(getattr(settings, "welcome_home_away_minutes", 30)),
        settle_minutes=int(getattr(settings, "welcome_home_settle_minutes", 20)),
        quiet_start_hour=int(getattr(settings, "owner_quiet_start_hour", 21)))
