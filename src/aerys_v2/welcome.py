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
     settle_minutes of getting home. D91 (Chris 11/22 17:11, "if she gives up maybe have it go
     through discord?"): when she gives up (he never settled in time, or something held her voice
     until HELLO_FOR), the hello goes to him as a Discord message instead, written for text. It
     carries nothing (the morning brief and anything held keep waiting for his room), and it stays
     quiet when he's been greeted another way (the brief just said it or is still due to; he's
     talked with her since he got home), or when he muted it. D96 (Chris 11/24): home after hours,
     the hello goes this way at once, as quiet text, never aloud. It's decided from durable
     records, so a restart up to GIVE_UP_GRACE past the hour decides the same way (a dropped row
     says why when nothing goes);
  3. not in quiet hours (a welcome home the next morning is wrong: it's dropped), and nothing
     holds her voice (a meeting or a nap: it waits, up to HELLO_FOR after he got home). The one
     exception, his D18 #3 (11/09, "ok"): home inside quiet hours with the morning brief still due,
     it's said in the brief's room, carrying the brief, the way the brief itself speaks there
     before 9 (both are brief rows at the desk). Coming home counts as being up. Settled anywhere
     else, it's dropped and the brief goes on its own when he reaches its room.
She composes it from facts only: how long he was out; what was on his personal calendars while
he was out (she asks how it went unless a check-in already asked, he muted appointments, or his
check-in dial is used up today); what was held for him; and the morning brief's facts when that's
still due; and (D85, 11/21: "yes please") where his phone stayed a while (his stops, from his
phone's own geocoded sensor over the trip), handed to her as facts rather than left for her to
go and find. It goes to the desk pinned to that room and outside the budget, like the briefs;
company mode turns it into a notification. If her words can't be had, a plain version goes.

What she says, what his calendar says, and where he went are the private part: never logged.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from aerys_v2.brief import Facts, _at, _csv, _ha, _loop, held_detail, plain_brief, report_house, stamped
from aerys_v2.followups import (CHECK_IN_TOPICS, CHECKIN_FOR, CHECKIN_LOCK, DATA_ONLY, HELLO_KEPT, Appointment,
                                appointment_key)
from aerys_v2.followups import SOURCE as FOLLOWUP_SOURCE
from aerys_v2.followups import TOPIC as FOLLOWUP_TOPIC
from aerys_v2.places import coords, places_near_for
from aerys_v2.reach import EASTERN, ReachEvent, local_day_start, muted_topics, next_local, told_quietly

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
# D88 (the owner, 11/22 10:31): he's greeted once. 11/22 10:02 he got "good morning" then "welcome home"
# 45 s apart: the brief was waiting for his room when he got home, and was said first. Now a brief still
# waiting is taken over by the hello (carried, one message); a brief already said within BRIEF_GREETED
# (or being delivered right now) means the hello stays quiet: he's been greeted.
BRIEF_GREETED = timedelta(minutes=15)
JUST_GREETED = "the morning brief just greeted him"
#: D91: why a hello given up on doesn't go to Discord either, and how long after the hour it still may
#: (a restart then still sends it; past this it's no longer a welcome).
BRIEF_GREETS = "the morning brief greets him when he reaches its room"
TALKED = "he's talked with her since he got home"
GIVE_UP_GRACE = timedelta(minutes=30)
ASK, NO_ASK = "ask", "don't ask"                 # how the facts mark what she may ask about
# D85 (his 11/21 20:17 "yes please"): his stops. On 11/21 the geocoded sensor changed about once a
# minute while he drove and held for 10 to 56 minutes where he stopped; at home and at a stop it
# flickers to a neighbour's address for a minute or three. A stop is a stay of STOP_MIN or more at
# one address, with interruptions of FLICKER or less folded back in. Addresses seen in HOME_WINDOW
# before he left, or at the moment he got home, are home's (and its neighbours'): never a stop.
STOP_MIN = timedelta(minutes=10)
NAMING_BUDGET_S = 8.0                           # D92: all of a trip's place look-ups together
NAMING_WORKERS = 4                              # D92: the look-ups' one shared pool
FLICKER = timedelta(minutes=5)
HOME_WINDOW = timedelta(hours=3)
MAX_STOPS = 6                                   # the longest few, told in the order he made them
PLACE_MAX = 120
#: D91: the hello she gives up on (he never reached a room with her speaker) goes here instead, as text.
TEXT_SURFACE = "discord"
TEXT_HELLO_FOR = timedelta(minutes=30)          # how long that message may wait at the desk (a meeting)

WELCOME_PROMPT = (
    "\n\n[Welcome home] Chris just got home after being out for a while, and the house has him "
    "settled in a room with you. Say hi, out loud: one to three short sentences, warm, first "
    "person, your own voice. If the facts list something from his calendar marked [ask], ask "
    "lightly how it went: one question. Never ask about anything marked [don't ask]: you may know "
    "he went, but you asked already, he'd rather you didn't, or it can wait. If nothing was on his calendar while he was out, you may "
    "ask one light, curious question about what he got up to. If the facts list where his phone "
    "stayed a while, you may mention where he went lightly and ask how it went, the way a friend "
    "who noticed would; when a stop lists named places, the one that fits how long he stayed and the time "
    "of day is most likely it: name that place itself, the way a friend who noticed would, and ask about it (\"a "
    "long lunch at the noodle place, then the hardware store?\"). Never mention addresses, maps or how you know; since "
    "it's a guess, put it as a question, not a claim; an address alone may need a quick look-up before "
    "you name the place. "
    "If the facts list what he told you himself, read that FIRST: when he told you, in his own voice, where "
    "he was going or what he'd be doing on THIS outing (said the same day, before he left or while he was "
    "out), that is where he was. Greet him like you remembered it (\"how was round two at her "
    "parents'?\") and never offer a place from the map instead; the map is only for when he told you "
    "nothing about this outing. A plan for another day, something he quoted or pasted from someone else, "
    "or a maybe doesn't count: when it isn't clearly about this outing, leave it and greet him plainly. "
    "Those lines are his private talk with you: say nothing from them aloud except the outing itself. "
    "Never guess: if it isn't clear, just ask. Never pry, and it's fine if he doesn't say. If things came in while he was out, say so in a few "
    "words; the details are in his messages. If his morning brief is included, give it after the "
    "hello in two to four more short sentences: the weather, what's left on today's calendar "
    "(work meetings: how many and when, never what they're about), anything that came in "
    "overnight. He's been up and out, so never talk as if he just woke up. Use ONLY the facts "
    "below (and what a look-up of one of his stops tells you) and never add any. Say times the way people say them (\"ten thirty\"). Nothing to read "
    "off a screen: no lists, no markdown, no emoji. Reply with ONLY what to say."
)

#: Without her own turn (no wake graph) her words come from a tool-free model call: nothing can be
#: looked up, so a street address never becomes a place name (Codex review of D85).
LOOKUP = "an address alone may need a quick look-up before you name the place. "
NO_LOOKUP = ("name a place only when the facts themselves say what it is (a business or park name), "
             "never from a street address alone. ")
LOOKUP_FACTS = " (and what a look-up of one of his stops tells you)"
WELCOME_PROMPT_TOOL_FREE = WELCOME_PROMPT.replace(LOOKUP, NO_LOOKUP).replace(LOOKUP_FACTS, "")

#: D91: the same hello, written as a Discord message because he never reached a room with her speaker.
SPOKEN_OPENING = ("Chris just got home after being out for a while, and the house has him settled in a room "
                  "with you. Say hi, out loud:")
TEXT_OPENING = ("Chris got home a while ago after being out, but he hasn't come by a room with your speaker, "
                "so this hello goes to him as a Discord message instead. Say hi in writing:")
SPOKEN_CLOSING = "Reply with ONLY what to say."
TEXT_CLOSING = "Reply with ONLY the message."
WELCOME_TEXT_PROMPT = WELCOME_PROMPT.replace(SPOKEN_OPENING, TEXT_OPENING).replace(SPOKEN_CLOSING, TEXT_CLOSING)
WELCOME_TEXT_PROMPT_TOOL_FREE = (WELCOME_PROMPT_TOOL_FREE.replace(SPOKEN_OPENING, TEXT_OPENING)
                                 .replace(SPOKEN_CLOSING, TEXT_CLOSING))


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
class Stop:
    place: str                        # the address his phone gave (data: never logged)
    start: datetime
    end: datetime
    at: tuple | None = None           # D92: where his phone sat (lat, lon), from the sensor's own location
    near: tuple = ()                  # D92: places.Place named there, nearest first (data: never logged)


@dataclass
class WelcomeFacts:
    away: timedelta
    room: str = ""
    visits: list = field(default_factory=list)   # Visit: his personal calendar while he was out
    held: list = field(default_factory=list)     # Reach rows held for him while he was out
    stops: list = field(default_factory=list)    # Stop: where his phone stayed a while (D85)
    brief: Facts | None = None                   # the morning brief's facts, when it comes along
    home_for: timedelta | None = None            # D91: set when it goes as a Discord message: how long he's been home
    told: list = field(default_factory=list)     # (when, his words): what he told her himself (kept out of the log)


def _utc(dt: datetime) -> datetime:
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def unattended_text_in(facts) -> bool:
    """Whether her turn must run fenced (no house writes): the facts carry a map's place names (D92), or
    his own typed lines, which can hold text he pasted from anywhere. In a DM he is there to see what she
    does with it; here nobody is watching, so it counts like any outside read."""
    return names_in(facts) or bool(getattr(facts, "told", None))


def names_in(facts) -> bool:
    """D92 (Codex review of b09f6db): the facts carry a map's place names, i.e. outside text: anyone can
    name a business. Her turn then runs as one that has read outside text (the wake fence refuses house
    writes), as after any outside read."""
    return any(getattr(s, "near", ()) for s in (getattr(facts, "stops", None) or []))


def _named(stop) -> str:
    """D92: the places named where his phone sat, as part of one line of data."""
    if not stop.near:
        return ""
    return "; named places there: " + "; ".join(f"{p.name} ({p.kind})" if p.kind else p.name for p in stop.near)


def _clock(dt: datetime) -> str:
    """'1:30 PM', Eastern (the calendar tool's format, without importing the tools)."""
    return dt.astimezone(EASTERN).strftime("%I:%M %p").lstrip("0")


def _day_clock(dt: datetime) -> str:
    """'Thu 3:14 PM': his own lines can come from the day before, so each carries its day."""
    return f"{_utc(dt).astimezone(EASTERN):%a} {_clock(dt)}"


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


def _place(state) -> str:
    """One address, as one line of data: whitespace collapsed, capped; a gap state is no place."""
    text = " ".join(str(state if state is not None else "").split())
    return "" if text.lower() in GAP_STATES else text[:PLACE_MAX]


def stops_from(history: list, left: datetime, arrived: datetime) -> list:
    """Where his phone stayed a while between `left` and `arrived` (D85), from the geocoded
    sensor's history (which may start before he left: that part tells home's addresses apart).
    Stays shorter than STOP_MIN don't count (driving, a red light); a FLICKER-or-shorter run of
    other addresses between two stays at the same one is folded into it; home's addresses
    (HOME_WINDOW before he left, and the one he got home to) never count. At most MAX_STOPS, the
    longest ones, in the order he made them."""
    left, arrived = _utc(left), _utc(arrived)
    rows = []
    for s in history or []:
        if not isinstance(s, dict):
            continue
        try:
            t = _utc(datetime.fromisoformat(str(s.get("last_changed")).replace("Z", "+00:00")))
        except ValueError:
            continue
        rows.append((t, _place(s.get("state")), coords((s.get("attributes") or {}).get("location"))
                     if isinstance(s.get("attributes"), dict) else None))
    rows.sort(key=lambda r: r[0])
    home = {p for t, p, _ in rows if left - HOME_WINDOW <= t < left}   # an address taken up AS he left is the trip's
    before_left = [p for t, p, _ in rows if t <= left - HOME_WINDOW]
    if before_left:
        home.add(before_left[-1])          # the address in effect as the window opened
    at_home = [p for t, p, _ in rows if t <= arrived]
    if at_home:
        home.add(at_home[-1])
    home.discard("")
    runs: list = []                        # [place, start, end, where], clipped to the trip
    for i, (t, place, where) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else arrived
        a, b = max(t, left), min(end, arrived)
        if b <= a:
            continue
        if runs and runs[-1][0] == place:
            runs[-1][2] = b
            runs[-1][3] = runs[-1][3] or where
        else:
            runs.append([place, a, b, where])
    merged: list = []
    for place, a, b, where in runs:
        between, folded = timedelta(0), False
        if place:
            for j in range(len(merged) - 1, -1, -1):
                if merged[j][0] == place:
                    merged[j][2] = b
                    merged[j][3] = merged[j][3] or where
                    del merged[j + 1:]
                    folded = True
                    break
                between += merged[j][2] - merged[j][1]
                if between > FLICKER:
                    break
        if not folded:
            merged.append([place, a, b, where])
    stops = [Stop(place=p, start=a, end=b, at=w) for p, a, b, w in merged
             if p and p not in home and b - a >= STOP_MIN]
    longest = sorted(stops, key=lambda s: s.end - s.start, reverse=True)[:MAX_STOPS]
    return sorted(longest, key=lambda s: s.start)


def welcome_facts_text(facts: WelcomeFacts) -> str:
    """The facts as the model reads them, framed as data: a calendar title can come from anyone
    who sends him an invite."""
    if facts.home_for is not None:
        lines = [f"He was out for {how_long(facts.away)} and got home {how_long(facts.home_for)} ago. He hasn't "
                 "come by a room with your speaker, so this goes to him as a Discord message."]
    else:
        lines = [f"He was out for {how_long(facts.away)}; he's in the {facts.room or 'house'} now."]
    if facts.visits:
        lines.append("On his calendar while he was out:")
        lines += [f"- [{ASK if v.ask else NO_ASK}] {v.when}: {v.title}" for v in facts.visits]
    else:
        lines.append("On his calendar while he was out: nothing")
    if facts.told:
        lines.append("What he told you himself in the hours before he left and while he was out (his own "
                     "words to you, oldest first, each with its day and time; text he pasted or quoted is "
                     "not him telling you his plans):")
        lines += [f"- {_day_clock(when)}: {words}" for when, words in facts.told]
    if facts.stops:
        lines.append("Where his phone stayed a while (addresses from his phone; an address isn't a place "
                     "name until it's looked up):")
        lines += [f"- {_clock(s.start)} to {_clock(s.end)}: {s.place}" + _named(s) for s in facts.stops]
        if any(s.near for s in facts.stops):
            lines.append("(Named places come from a map look-up around where his phone sat, nearest first, then "
                         "the best-known close by: he was most likely at one of them, but not certainly.)")
    if facts.held:
        lines.append("Came in while he was out (the details are in his messages):")
        lines += [f"- {stamped(r)}" for r in facts.held]
    b = facts.brief
    if b is not None:
        lines.append("His morning brief (still due today):")
        lines.append(f"- Weather today: {b.weather or 'unknown (say nothing about it)'}")
        lines.append("- Left on today's calendar: " + ("; ".join(b.events) if b.events else "nothing"))
        overnight = [*b.held, *b.quiet]
        lines.append("- Came in overnight, each with the time it happened: "
                     + ("; ".join(stamped(r) for r in overnight) if overnight
                        else "nothing (say nothing about it)"))
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


#: How long before the brief's cutoff a waiting hello lets the brief go on its own: its 60 s poll
#: plus the time it takes to compose and speak.
DEFER_MARGIN = timedelta(minutes=5)


class WelcomeHome:
    """tick() -> what happened this pass (logged, never the words). Every leg is injected:
    read() -> HomeReading; away_since(arrived) -> when he left, or None (not a trip);
    visits(left, arrived) -> the Appointments on his personal calendars in between;
    compose(WelcomeFacts) -> her words; stops(left, arrived) -> where his phone stayed a while
    (D85), or None (not armed). brief: the MorningBrief, when armed (its lock is shared,
    so the two never compose at once). followups: the follow-up store, when armed."""

    def __init__(self, desk, *, read: Callable[[], HomeReading],
                 away_since: Callable[[datetime], datetime | None],
                 compose: Callable[[WelcomeFacts], str],
                 visits: Callable[[datetime, datetime], list] | None = None,
                 stops: Callable[[datetime, datetime], list] | None = None,
                 rooms: tuple = ("office", "kitchen"), brief=None, followups=None,
                 away_minutes: int = 30, settle_minutes: int = 20, quiet_start_hour: int = 21,
                 clock: Callable[[], datetime] | None = None,
                 compose_text: Callable[[WelcomeFacts], str] | None = None,
                 told: Callable[[datetime, datetime], list] | None = None):
        self.desk, self.read, self.away_since, self.compose, self.visits = desk, read, away_since, compose, visits
        self.compose_text = compose_text or compose     # D91: her words for the Discord hello
        self.stops = stops
        self.told = told              # told(left, arrived) -> [(when, his words)], or None (not armed)
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
        thread: if the arrivals change under it, it says wait (the next pass asks again).
        Never within DEFER_MARGIN of the brief's cutoff: past that the brief goes on its own, so a
        hello still waiting can't swallow it (Codex, 11/09: arriving 12:50 unsettled, nothing was
        said by 1 PM)."""
        b = self.brief
        if b is not None:
            if now >= _at(now.astimezone(EASTERN).date(), b.cutoff_hour) - DEFER_MARGIN:
                return False
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
            if now - arrived <= HELLO_FOR + GIVE_UP_GRACE:
                return self._give_up(key, arrived, now)                  # D91: the hour ended unsaid
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
        if TOPIC in muted_topics(store.prefs(self.desk.budget_default), now):
            self._drop(key, now, f"you muted {TOPIC}")
            self._finish(key)
            return "muted"
        # Inside quiet hours only a hello that carries the morning brief is said, in the brief's room.
        # D96 (Chris 11/24 23:05, "I dont want her blowing the house up after hours in voice but I would
        # have appreciated a welcome home thing from her tonight"): any other hello after hours goes to
        # him as a Discord text right away, never aloud, with D91's reasons to stay quiet still standing.
        night = self.desk.quiet(now)
        if night and not (self.brief is not None and self._brief_due(now, now.astimezone(EASTERN).date())):
            return self._give_up(key, arrived, now)
        if not (r.room in self.rooms and r.confirmed):
            if now - arrived > self.settle_for:
                return self._give_up(key, arrived, now)                  # D91: never settled in time
            return "waiting to settle"
        if night and r.room != self.brief.room:
            self._drop(key, now, "quiet hours")
            self._finish(key)
            return "quiet hours"
        held_by = self.desk.voice_held(now)
        if held_by:
            return f"held: {held_by}"
        stops = self._stops(left, arrived)     # before the brief's lock: naming them can take seconds (D92)
        with self.lock:
            return self._say(key, arrived, left, r.room, now, must_carry=night, stops=stops)

    def _give_up(self, key: str, arrived: datetime, now: datetime) -> str:
        """D91 (Chris 11/22 17:11, "if she gives up maybe have it go through discord?"): the spoken hello's
        window closed unsaid (he never settled in a room with her speaker, or something held her voice
        the whole hour), so it goes to him as a Discord message instead of nowhere. Everything it asks is
        durable, so a restart decides the same way (Codex review of 1e42a9c): what the desk already holds
        under this key, the trip from HA's own history. It stays quiet when he's been greeted another way:
        the morning brief just said it, the brief is still due (it greets him when he reaches its room:
        one greeting, D88), or he's talked with her since he got home. Never muted. After hours (D96) it goes
        too, as quiet text: the only after-hours path to a spoken hello is the brief's (D18 #3). A read
        that fails waits for the next pass (inside GIVE_UP_GRACE), never guesses."""
        store = self.desk.store
        day = now.astimezone(EASTERN).date()
        try:
            if store.decided(key, arrived - timedelta(minutes=1)):      # said, dropped or texted already
                self._finish(key)
                return "done"
            if key not in self.left:                    # a restart: the trip from HA's own history
                self.left[key] = self.away_since(arrived)
                self.arrived[key] = arrived
        except Exception as e:
            log.warning("welcome home: couldn't check the arrival (%s); trying again", type(e).__name__)
            return "waiting: couldn't check the arrival"
        left = self.left[key]
        if left is None or arrived - left < self.away_for:
            self._finish(key)
            return "not a trip" if left is None else "a short trip"
        # D96: after hours is no longer a reason to stay quiet; it goes as text (the desk sends a text-only
        # hello at night as quiet text). Before nine with the brief due, _why_quiet keeps it for the brief.
        # A first look without the lock, so a give-up that stays quiet costs no place look-ups (D92); then
        # the stops, named outside the brief's lock (it can take seconds); then the deciding look and the
        # send under it. The brief's own thread speaks only under that lock, so it can't greet him between
        # them (Codex review of c288eee: a brief sent mid-check left him greeted twice).
        quiet = self._quiet_now(key, now, day, arrived)      # a first look: a quiet give-up costs no look-ups
        if quiet is not None:
            return quiet
        stops = self._stops(left, arrived)
        with self.lock:
            quiet = self._quiet_now(key, now, day, arrived)  # the deciding look, under the lock, then the send
            if quiet is not None:
                return quiet
            outcome = self._say(key, arrived, left, "", now, by_text=True, stops=stops)
        return f"by text: {outcome}"

    def _quiet_now(self, key: str, now: datetime, day: date, arrived: datetime) -> str | None:
        """The give-up's look at whether to stay quiet: the reason (recorded, and the arrival finished),
        a wait when the look failed, or None to go ahead."""
        try:
            why = self._why_quiet(now, day, arrived)
        except Exception as e:
            log.warning("welcome home: couldn't check whether he's been greeted (%s); trying again",
                        type(e).__name__)
            return "waiting: couldn't check whether he's been greeted"
        if why is not None:
            self._drop(key, now, why)
            self._finish(key)
        return why

    def _why_quiet(self, now: datetime, day: date, arrived: datetime) -> str | None:
        """D91: why a hello given up on doesn't go by Discord either, or None. A failed read raises."""
        if TOPIC in muted_topics(self.desk.store.prefs(self.desk.budget_default), now):
            return f"you muted {TOPIC}"
        brief = self._brief_greets(now, day)
        if brief is not None:
            return brief
        replied = getattr(self.desk, "replied_since", None)
        return TALKED if replied is not None and replied(arrived) else None

    def _brief_just_greeted(self, now: datetime, brief_key: str) -> bool:
        """Today's brief was SAID (delivered, not just queued: a brief can wait hours for his room)
        within BRIEF_GREETED: he's been greeted. A failed read raises: the caller waits, never guesses."""
        rows = self.desk.store.rows_for(brief_key)
        return any(r.status == "sent" and r.delivered_at is not None and now - _utc(r.delivered_at) <= BRIEF_GREETED
                   for r in rows)

    def _brief_greets(self, now: datetime, day: date) -> str | None:
        """D91: has today's brief greeted him, or will it in its room? From ONE read of its rows, so a
        delivery (queued -> sent, under the desk's lock) can't slip between two reads; called under
        self.lock, so the brief's own thread can't put a new one out meanwhile (Codex review of c288eee).
        JUST_GREETED (said within BRIEF_GREETED), BRIEF_GREETS (queued in its room, or still due), or None.
        A failed read raises: the caller waits, never guesses."""
        if self.brief is None:
            return None
        rows = self.desk.store.rows_for(f"brief:{day.isoformat()}")
        if any(r.status == "sent" and r.delivered_at is not None and now - _utc(r.delivered_at) <= BRIEF_GREETED
               for r in rows):
            return JUST_GREETED
        if any(r.status == "queued" for r in rows):
            return BRIEF_GREETS
        if any(r.status == "sent" for r in rows):
            return None                     # it went earlier today: nothing still to come
        return BRIEF_GREETS if self._brief_due(now, day) else None

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
        return "brief" not in muted_topics(store.prefs(self.desk.budget_default), now)

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

    def _stops(self, left: datetime, arrived: datetime) -> list:
        """Where his phone stayed a while (D85). A read that fails costs only the stops: the hello
        goes without them, and nothing about where he went is logged (the error's type only)."""
        if self.stops is None:
            return []
        try:
            return list(self.stops(left, arrived) or [])
        except Exception as e:
            log.warning("welcome home: his stops couldn't be read (%s); going without them", type(e).__name__)
            return []

    def _told(self, left: datetime, arrived: datetime) -> list:
        """What he told her himself that day (11/26: he'd said "when you see me out again its because i was
        at her parents", and the hello asked about a place off the map). A read that fails costs only
        these lines; none of his words are logged (the error's type only)."""
        if self.told is None:
            return []
        box: dict = {}

        def work():
            try:
                box["rows"] = list(self.told(left, arrived) or [])
            except Exception as e:
                box["err"] = type(e).__name__

        # He's standing in the room: a read that hangs never holds the hello (it goes without these lines).
        worker = threading.Thread(target=work, daemon=True, name="welcome-told")
        worker.start()
        worker.join(TOLD_BUDGET_S)
        if "rows" in box:
            return box["rows"]
        log.warning("welcome home: what he told her couldn't be read (%s); going without it",
                    box.get("err") or "took too long")
        return []

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
        may_ask = "appointments" not in muted_topics(prefs, now)
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
            # No claim store: nothing can mark an appointment asked, so two trips during one long
            # appointment would each ask (Codex, 11/09). Not asked at all; the hello still goes.
            may_ask = False
        chosen: dict = {}
        for a in sorted(found, key=lambda a: a.end, reverse=True):
            if not may_ask:
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
        if "appointments" in muted_topics(prefs, now):
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
             must_carry: bool = False, by_text: bool = False, stops: list | None = None) -> str:
        day = now.astimezone(EASTERN).date()
        # Never said at night, and not long after he got home (the desk drops it past this).
        expires = min(arrived + HELLO_FOR, next_local(arrived, self.quiet_start_hour))
        if by_text:
            # D91: given up on as a spoken hello, it goes now as a message. D96: after hours too, so its
            # window is just TEXT_HELLO_FOR (the desk sends a text-only hello at night as quiet text).
            expires = now + TEXT_HELLO_FOR
        store = self.desk.store
        took = False
        # D91: a Discord hello carries nothing: the brief (made to be said) keeps waiting for his room.
        if self.brief is not None and not by_text:
            brief_key = f"brief:{day.isoformat()}"
            try:
                outcome = self.desk.take_over_brief(brief_key, now, _at(day, self.brief.start_hour))
                greeted = outcome == "delivering" or (outcome in ("went", "free")
                                                      and self._brief_just_greeted(now, brief_key))
            except Exception as e:
                # Never on an uncertain answer: the brief may still go, or may just have gone, so the
                # hello waits and asks again next pass (Codex reviews of 5bdaab9 and 73acf79).
                log.warning("welcome home: couldn't check the morning brief (%s); trying again", type(e).__name__)
                return "waiting: couldn't check the morning brief"
            if greeted:
                self._drop(key, now, JUST_GREETED)
                self._finish(key)
                return "the brief just greeted him"
            if outcome == "taken":
                # The brief process counts today as handed off; reopened, its own tick waits while this
                # hello carries it and goes on its own if the hello never does (store.carried).
                self.brief.done = None
                took = True
                log.info("welcome home: took over the waiting morning brief, one greeting")
        brief_facts, marker = None, None
        carrying = not by_text and (took or self._brief_due(now, day))
        if carrying:
            # The marker goes first, and the hello carries it in its folds: the brief counts as given
            # only while that hello is queued or said (store.carried), so a hello that never goes, or
            # a restart before it's written, leaves the brief free to go on its own.
            try:
                marker = store.add(ReachEvent(source="brief", topic="brief", headline="(carried by the welcome-home hello)",
                                              brief=True, dedupe_key=f"brief:{day.isoformat()}"), now, status="folded",
                                   reason="carried by the welcome-home hello", not_before=None)
            except Exception:
                log.warning("welcome home: couldn't mark the brief as carried; it goes on its own", exc_info=True)
        if carrying and marker is None:
            # Meant to carry the brief but it couldn't be marked carried: the brief (reopened if it was
            # taken over) goes on its own, as store.carried sees no carrier, so the hello stays quiet
            # rather than greet him twice (Codex review of 73acf79: the same on the brief-due path).
            self._drop(key, now, "couldn't carry the morning brief; the brief goes on its own")
            self._finish(key)
            return "the brief goes on its own"
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
            quiet = told_quietly(store.sent_quietly(since))
            try:
                brief_facts = self.brief.gather(now, held, quiet)
            except Exception:
                log.warning("welcome home: the brief's day couldn't be gathered; going with what she holds",
                            exc_info=True)
                brief_facts = Facts(held=held, quiet=quiet)
        elif by_text:
            held = []                    # D91: what was held keeps waiting for his room, as it would have
        else:
            held = self.desk.reserve_for_brief(left, expires, reason="carried by the welcome-home hello",
                                               skip_topics=CHECK_IN_TOPICS)
        visits = self._visits(left, arrived, now, key)
        facts = WelcomeFacts(away=arrived - left, room=room, visits=visits, held=[] if marker else held,
                             stops=self._stops(left, arrived) if stops is None else stops, brief=brief_facts,
                             home_for=(now - arrived) if by_text else None, told=self._told(left, arrived))
        compose = self.compose_text if by_text else self.compose
        text = ""
        try:
            text = (compose(facts) or "").strip()
        except Exception as e:      # the type only: her facts (his calendar, where he went) can be in the text
            log.warning("welcome home: her words failed (%s); the plain version goes", type(e).__name__)
        if not text:
            text = plain_welcome(facts)
        folds = tuple(h.id for h in held) + ((marker.id,) if marker is not None else ())
        report_house(self.desk, getattr(compose, "house", None), source=SOURCE, topic=TOPIC, key=key)
        if marker is not None and brief_facts is not None:
            # D101 (Codex 11/25): the hello carries the brief, so it carries what she closed too, under the
            # brief's own key (the brief itself won't run again today).
            from aerys_v2.brief import tell_closed_promises

            tell_closed_promises(self.desk, brief_facts, getattr(self.brief, "promises_told", None),
                                 key=f"brief:{day.isoformat()}")
        outcome = self.desk.reach_out(ReachEvent(
            source=SOURCE, topic=TOPIC, headline=text, detail=held_detail(held), room=None if by_text else room,
            brief=True, folds=folds, dedupe_key=key, expires_at=expires, text_only=by_text,
            text_to=TEXT_SURFACE if by_text else None))
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
        log.info("welcome home: %s%s (%d held, %d on the calendar, brief %s)", outcome,
                 " by Discord (he never reached a speaker room)" if by_text else "", len(held), len(visits),
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


#: How far before he left his own words are read, how many lines, and how long each may be.
TOLD_WINDOW = timedelta(hours=10)
TOLD_LINES = 30
TOLD_CHARS = 240
TOLD_BUDGET_S = 6.0
#: Only where he talks to her alone: his own thread, by DM, by voice or the panel.
TOLD_CHANNELS = ("discord_dm", "voice", "person", "telegram")

TOLD_SQL = """
SELECT created_at, input_text FROM v2_turns
 WHERE person_id = %(owner)s AND thread_id = %(thread)s AND channel = ANY(%(channels)s)
   AND sealed_room IS NULL AND created_at >= %(start)s AND created_at <= %(end)s
   AND input_text IS NOT NULL AND btrim(input_text) <> ''
 ORDER BY created_at DESC LIMIT %(n)s
"""


def told_rows(rows, *, lines: int = TOLD_LINES, chars: int = TOLD_CHARS) -> list:
    """(when, his words) oldest first, each on one line and clipped; the newest `lines` are kept."""
    out = []
    for when, text in rows or []:
        words = " ".join(str(text or "").split())
        if not words or not isinstance(when, datetime):
            continue
        out.append((when, words if len(words) <= chars else words[:chars - 1].rstrip() + "…"))
    out.sort(key=lambda r: r[0])
    return out[-lines:]


def told_for(settings, *, connect=None) -> Callable[[datetime, datetime], list] | None:
    """What he told her himself, from his own private thread (her own conversation with him: DMs, voice,
    the panel), from TOLD_WINDOW before he left to when he got home. None without a database or an
    owner. Never another person's words, never a shared or sealed room. connect is the test seam."""
    url = getattr(settings, "database_url", None)
    owner = str(getattr(settings, "owner_person_id", "") or "").strip()
    if not owner or (not url and connect is None):
        return None
    if connect is None:
        from aerys_v2.factory import read_only_db

        connect = read_only_db(url)

    def read(left: datetime, arrived: datetime) -> list:
        with connect() as conn:
            rows = conn.execute(TOLD_SQL, {"owner": owner, "thread": f"person:{owner}",
                                           "channels": list(TOLD_CHANNELS), "start": left - TOLD_WINDOW,
                                           "end": arrived, "n": TOLD_LINES}).fetchall()
        return told_rows(rows)

    return read


def stops_for(settings, *, get=None, near=None) -> Callable[[datetime, datetime], list] | None:
    """D85: where his phone stayed a while, from HA's recorder for his phone's geocoded sensor
    (presence_geocoded_entity: geocoded on the phone), read from HOME_WINDOW before he left (to tell
    home's addresses apart) to when he got home. None when no geocoded sensor is configured.
    D92: with the sensor's attributes (its `location`), and each stop named from Google Places
    (places.places_near_for, when its key is set): the look-ups run together, bounded by
    NAMING_BUDGET_S; one that fails or runs late costs only that stop's names. get(url, headers) and
    near(lat, lon) are the test seams."""
    import httpx
    from urllib.parse import quote, urlencode

    entity = str(getattr(settings, "presence_geocoded_entity", "") or "").strip()
    if not entity:
        return None
    base, headers = _ha(settings)
    fetch = get or (lambda url, hdrs: httpx.get(url, headers=hdrs, timeout=10.0))

    def read(left: datetime, arrived: datetime) -> list:
        start = left - HOME_WINDOW
        url = (f"{base}/api/history/period/{quote(start.isoformat())}?"
               + urlencode({"filter_entity_id": entity, "end_time": arrived.isoformat()}))
        r = fetch(url, headers)
        r.raise_for_status()
        data = r.json()
        rows = data[0] if isinstance(data, list) and data and isinstance(data[0], list) else []
        return named(stops_from(rows, left, arrived), namer)

    namer = near if near is not None else places_near_for(settings)
    return read


def named(stops: list, near) -> list:
    """D92: each stop with the places named where his phone sat. All at once, NAMING_BUDGET_S in all; a
    look-up that fails or is still running then costs only its stop's names (logged by type, never where)."""
    if near is None or not stops:
        return stops
    import concurrent.futures as cf

    todo = [(i, s) for i, s in enumerate(stops) if s.at is not None]
    if not todo:
        return stops
    out = list(stops)
    jobs = {_naming_pool().submit(near, *s.at): i for i, s in todo}
    done, late = cf.wait(jobs, timeout=NAMING_BUDGET_S)
    for job in done:
        try:
            found = tuple(job.result() or ())
        except Exception as e:
            log.warning("welcome home: a stop couldn't be named (%s); its address goes alone", type(e).__name__)
            continue
        i = jobs[job]
        out[i] = replace(out[i], near=found)
    if late:
        for job in late:
            job.cancel()                # one that never started never runs; a running one ends at its timeouts
        log.warning("welcome home: %d stop(s) not named in time; their addresses go alone", len(late))
    return out


_POOL: list = []
_POOL_LOCK = threading.Lock()


def _naming_pool():
    """One small pool for every look-up (Codex review of b09f6db): a look-up that outlives its budget keeps
    one of NAMING_WORKERS threads until its own timeouts end it, never a new thread per welcome."""
    import concurrent.futures as cf

    with _POOL_LOCK:
        if not _POOL:
            _POOL.append(cf.ThreadPoolExecutor(max_workers=NAMING_WORKERS, thread_name_prefix="stop-names"))
        return _POOL[0]


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


def welcome_home_for(settings, desk, *, brief=None, action_graph=None) -> WelcomeHome | None:
    """None unless armed: welcome_home on, its entity, a desk, HA, and a speaker in one of its rooms."""
    entity = str(getattr(settings, "welcome_home_entity", "") or "").strip()
    if (not getattr(settings, "welcome_home", False) or not entity or desk is None
            or getattr(settings, "ha_token", None) is None):
        return None
    from aerys_v2.brief import composer_for
    from aerys_v2.kael_line import parse_room_devices
    from aerys_v2.wakeups import her_turn_for

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
    turn = her_turn_for(settings, action_graph)
    log.info("welcome home armed | %s", "her own turn" if turn else "tool-free words (no wake graph)")
    return WelcomeHome(
        desk, read=home_reader_for(settings), away_since=away_since_for(settings),
        compose=composer_for(settings, prompt=WELCOME_PROMPT if turn else WELCOME_PROMPT_TOOL_FREE,
                             text_of=welcome_facts_text, turn=turn,
                             kind="welcome", outside_of=unattended_text_in),
        compose_text=composer_for(settings, prompt=WELCOME_TEXT_PROMPT if turn else WELCOME_TEXT_PROMPT_TOOL_FREE,
                                  text_of=welcome_facts_text, turn=turn, kind="welcome-text", outside_of=unattended_text_in),
        visits=visits_for(settings), stops=stops_for(settings), told=told_for(settings),
        rooms=rooms, brief=brief, followups=followups,
        away_minutes=int(getattr(settings, "welcome_home_away_minutes", 30)),
        settle_minutes=int(getattr(settings, "welcome_home_settle_minutes", 20)),
        quiet_start_hour=int(getattr(settings, "owner_quiet_start_hour", 21)))
