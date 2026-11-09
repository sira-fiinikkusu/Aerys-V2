"""Her morning brief: spoken in the office, the first time the house places him there after
he's up (board #176, Stage 2).

The owner, 2026-11-08: Kael's HA brief stays on his phone and watch ("since hers is over
voice and yours is on my phone/watch it is extremely helpful because im groggy af till i
have coffee in me"), and "her voice morning brief hits in the office when she first detects
me in there after waking ... this way im more awake and it doesnt impact megan".

Once a day, from START until CUTOFF (Eastern), when all of these hold:
  1. the house has settled him in the brief's room, and motion there backs his phone up
     (the settled room's 'confirmed': a phone left charging there overnight is not him);
  2. he's up: one of the wake signals moved this morning (Kael's HA brief fired, his watch
     came on) or he's talked to her since START; from SOLO on, the room alone will do;
  3. nothing holds her voice (a meeting, a nap).
She composes it from facts only (the day's weather, what's left on today's calendar with
work meetings as times only, what she held or sent quietly overnight) and hands it to the
desk pinned to that room, outside the budget, carrying the items held overnight so they
don't also go out one by one. If her words can't be had, a plain version goes. Past CUTOFF
there's no brief; anything still held then goes on its own.

Her evening heads-up (EveningHeadsUp, below) is the brief's short sibling: once an evening,
before quiet hours, the first time the house places him in a room with a speaker that isn't
the bedroom, a sentence or three about TOMORROW, and only when tomorrow has something worth
the words (an early start, something personal on the calendar, rain). A quiet tomorrow gets
no heads-up at all: the owner's word for the brief he liked was "not noisy".

What she says is the private part: never logged.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable

from aerys_v2.followups import CHECK_IN_TOPICS
from aerys_v2.reach import EASTERN, Reach, ReachEvent

log = logging.getLogger(__name__)

TICK_S = 60
SOURCE = "brief"
TOPIC = "brief"          # he can mute it in words, like any topic (tools/reach_prefs.py)

BRIEF_PROMPT = (
    "\n\n[Your morning brief] Chris just walked into his office. He's up, but groggy until "
    "he's had coffee. Give him his morning brief out loud: two to five short sentences, warm, "
    "first person, your own voice, the most useful thing first. Use ONLY the facts below and "
    "never add any: no weather, plans or news that aren't listed. Say times the way people say "
    "them (\"ten thirty\"). Work meetings: how many and when, never what they're about. If "
    "something came in overnight, say so in a few words; the details are in his messages. "
    "Nothing to read off a screen: no lists, no markdown, no emoji. Reply with ONLY what to say."
)


@dataclass
class Reading:
    room: str = ""                  # the settled room, lowercased ('' when unknown or away)
    confirmed: bool = False         # motion in that room backs his phone up
    woke_at: datetime | None = None # the latest wake signal this morning, if any


@dataclass
class Facts:
    weather: str = ""
    events: list = field(default_factory=list)    # "10:00 AM to 10:30 AM: a work meeting", ...
    held: list = field(default_factory=list)      # Reach rows held overnight: the brief carries them
    quiet: list = field(default_factory=list)     # Reach rows already sent as quiet text overnight
    first_at: datetime | None = None              # the day's first timed event
    personal: int = 0                             # events from his personal calendars
    wet: bool = False                             # rain (or worse) in the forecast
    complete: bool = True                         # every source answered (else "quiet" may be a gap)


def _at(day: date, hour: int) -> datetime:
    return datetime.combine(day, time(hour), tzinfo=EASTERN).astimezone(timezone.utc)


def plain_brief(facts: Facts) -> str:
    """The brief without her words (the model failed or answered nothing): the facts, plainly."""
    parts = ["Good morning."]
    if facts.weather:
        parts.append(f"Today: {facts.weather.rstrip('.')}.")
    if facts.events:
        when, _, what = facts.events[0].partition(": ")
        first = f"{what} at {when.split(' to ')[0]}" if what and not when.startswith("all day") else facts.events[0]
        more = len(facts.events) - 1
        parts.append(f"First up, {first}" + (f", and {more} more after that." if more else "."))
    else:
        parts.append("Nothing on the calendar for the rest of today.")
    overnight = len(facts.held) + len(facts.quiet)
    if overnight == 1:
        parts.append("One thing came in overnight; it's in your messages.")
    elif overnight:
        parts.append(f"{overnight} things came in overnight; they're in your messages.")
    return " ".join(parts)


def facts_text(facts: Facts) -> str:
    """The facts as the model reads them."""
    lines = [f"Weather today: {facts.weather or 'unknown (say nothing about it)'}"]
    lines.append("Left on today's calendar:" if facts.events else "Left on today's calendar: nothing")
    lines += [f"- {e}" for e in facts.events]
    if facts.held or facts.quiet:
        lines.append("Came in overnight (the details are in his messages):")
        lines += [f"- {r.headline}" for r in [*facts.held, *facts.quiet]]
    else:
        lines.append("Came in overnight: nothing")
    return "\n".join(lines)


def held_detail(held: list) -> str:
    """What the brief carries as text: the items held overnight, in full (detail is never spoken)."""
    if not held:
        return ""
    items = [f"- {r.headline}" + (f"\n  {r.detail}" if r.detail else "") for r in held]
    return "Held for you overnight:\n" + "\n".join(items)


class MorningBrief:
    """tick() -> what happened this pass (logged, never the words). Every leg is injected."""

    def __init__(self, desk, *, read: Callable[[], Reading],
                 gather: Callable[[datetime, list, list], Facts],
                 compose: Callable[[Facts], str],
                 replied_since: Callable[[datetime], bool] | None = None,
                 room: str = "office", start_hour: int = 5, solo_hour: int = 8, cutoff_hour: int = 13,
                 clock: Callable[[], datetime] | None = None):
        if not (0 <= start_hour <= solo_hour <= cutoff_hour <= 24):
            raise ValueError("brief hours must run start <= solo <= cutoff")
        self.desk, self.read, self.gather, self.compose = desk, read, gather, compose
        self.replied_since = replied_since
        self.room = room.strip().lower()
        self.start_hour, self.solo_hour, self.cutoff_hour = start_hour, solo_hour, cutoff_hour
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.done: date | None = None   # the day this process already handed a brief to the desk
        # Shared with the welcome-home hello (welcome.py), which may carry the brief: one of them
        # decides and composes at a time, so the brief is never said twice.
        self.lock = threading.Lock()
        # defer(now) -> True while the welcome-home hello waits for him to settle after a trip (welcome.py
        # sets it): the hello carries the brief, so the brief doesn't race it to his room.
        self.defer: Callable[[datetime], bool] | None = None

    def _up(self, reading: Reading, since: datetime) -> bool:
        if reading.woke_at is not None and reading.woke_at >= since:
            return True
        if self.replied_since is None:
            return False
        try:
            return self.replied_since(since)
        except Exception:
            log.warning("brief: reply check failed; not counting it as up", exc_info=True)
            return False

    def tick(self) -> str:
        with self.lock:
            return self._tick()

    def _tick(self) -> str:
        now = self.clock()
        local = now.astimezone(EASTERN)
        day = local.date()
        if not (self.start_hour <= local.hour < self.cutoff_hour):
            return "outside the window"
        if self.done == day:
            return "done"
        start, cutoff = _at(day, self.start_hour), _at(day, self.cutoff_hour)
        key = f"brief:{day.isoformat()}"
        store = self.desk.store
        if store.recently_sent(key, start):           # given (or waiting on a busy room) before a restart
            self.done = day
            return "done"
        # Carried by the welcome-home hello (welcome.py): done once that's said; while it waits
        # for its room the brief waits too; if it never goes, the brief is free to go on its own.
        carried = store.carried(key, start)
        if carried == "sent":
            self.done = day
            return "done"
        if carried == "pending":
            return "carried by the welcome-home hello"
        if TOPIC in {str(m).strip().lower() for m in store.prefs(self.desk.budget_default).muted}:
            return "muted"
        try:
            deferring = self.defer is not None and self.defer(now)
        except Exception:
            log.warning("brief: couldn't ask the welcome-home hello; going on", exc_info=True)
            deferring = False
        if deferring:
            return "waiting for the welcome-home hello"
        try:
            reading = self.read()
        except Exception:
            log.warning("brief: can't read the house this pass", exc_info=True)
            return "no reading"
        if reading.room != self.room or not reading.confirmed:
            return f"not in the {self.room}"
        if local.hour < self.solo_hour and not self._up(reading, start):
            return "not up yet"
        held_by = self.desk.voice_held(now)
        if held_by:
            return f"held: {held_by}"
        since = _at(day - timedelta(days=1), 18)       # "overnight": since yesterday evening
        # Reserved first: what it will carry can't also go out on its own while she composes.
        # Never a check-in (a question with its own life, followups.py): it goes on its own.
        held = self.desk.reserve_for_brief(since, cutoff, skip_topics=CHECK_IN_TOPICS)
        quiet = store.sent_quietly(since)
        quiet = [r for r in quiet if not r.text_only]   # never said out loud, not even in the brief (email)
        try:
            facts = self.gather(now, held, quiet)
        except Exception:
            log.warning("brief: gathering the day failed; going with what she holds", exc_info=True)
            facts = Facts(held=held, quiet=quiet)
        text = ""
        try:
            text = (self.compose(facts) or "").strip()
        except Exception:
            log.warning("brief: her words failed; the plain version goes", exc_info=True)
        if not text:
            text = plain_brief(facts)
        outcome = self.desk.reach_out(ReachEvent(
            source=SOURCE, topic=TOPIC, headline=text, detail=held_detail(held), room=self.room,
            brief=True, folds=tuple(r.id for r in held), dedupe_key=key, expires_at=cutoff))
        self.done = day        # whatever the desk said, once a day: never a model call a minute
        log.info("brief %s: %s (%d held, %d quiet)", day, outcome, len(held), len(quiet))
        return outcome

    def run_forever(self) -> None:  # pragma: no cover - thin loop over tick()
        log.info("morning brief armed | %s, %02d:00-%02d:00", self.room, self.start_hour, self.cutoff_hour)
        _loop(self.tick, "brief")


def _loop(tick: Callable[[], str], name: str) -> None:  # pragma: no cover - thin loop
    pause = threading.Event()
    while True:
        try:
            tick()
        except Exception:
            log.warning("%s tick failed", name, exc_info=True)
        pause.wait(TICK_S)


def start_morning_brief(brief: "MorningBrief | None") -> threading.Thread | None:
    if brief is None:
        return None
    thread = threading.Thread(target=brief.run_forever, daemon=True, name="morning-brief")
    thread.start()
    return thread


# --- the evening heads-up --------------------------------------------------------------------

EVENING_SOURCE = "evening"
EVENING_TOPIC = "evening"

EVENING_PROMPT = (
    "\n\n[Your evening heads-up] It's evening and Chris is home. Give him a heads-up about "
    "TOMORROW, out loud: one to three short sentences, warm, first person, your own voice, the "
    "thing he'd most want to know first (an early start, something personal on the calendar, "
    "rain). Use ONLY the facts below and never add any. Say times the way people say them. Work "
    "meetings: how many and when, never what they're about. Nothing to read off a screen: no "
    "lists, no markdown, no emoji. Reply with ONLY what to say."
)


def worth_saying(facts: Facts, *, early_hour: int = 10) -> bool:
    """Tomorrow earns a heads-up with an early start, anything personal, or rain."""
    early = facts.first_at is not None and facts.first_at.astimezone(EASTERN).hour < early_hour
    return early or facts.personal > 0 or facts.wet


def plain_evening(facts: Facts) -> str:
    parts = ["Heads-up for tomorrow:"]
    if facts.events:
        when, _, what = facts.events[0].partition(": ")
        parts.append((f"{what} at {when.split(' to ')[0]}" if what and not when.startswith("all day")
                      else facts.events[0]) + (f", and {len(facts.events) - 1} more." if len(facts.events) > 1 else "."))
    if facts.weather:
        parts.append(f"Weather: {facts.weather.rstrip('.')}.")
    return " ".join(parts)


def evening_facts_text(facts: Facts) -> str:
    lines = [f"Tomorrow's weather: {facts.weather or 'unknown (say nothing about it)'}"]
    lines.append("Tomorrow's calendar:" if facts.events else "Tomorrow's calendar: nothing")
    lines += [f"- {e}" for e in facts.events]
    return "\n".join(lines)


class EveningHeadsUp:
    """tick() -> what happened this pass. Once an evening, in a room with a speaker that isn't
    the bedroom, before quiet hours, and only when tomorrow is worth a heads-up."""

    def __init__(self, desk, *, read: Callable[[], Reading], gather: Callable[..., Facts],
                 compose: Callable[[Facts], str], rooms: tuple = ("office", "kitchen"),
                 start_hour: int = 19, quiet_start_hour: int = 21, early_hour: int = 10,
                 clock: Callable[[], datetime] | None = None):
        self.desk, self.read, self.gather, self.compose = desk, read, gather, compose
        self.rooms = tuple(r.strip().lower() for r in rooms if r.strip() and r.strip().lower() != "bedroom")
        self.start_hour, self.quiet_start_hour, self.early_hour = start_hour, quiet_start_hour, early_hour
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.done: date | None = None

    def tick(self) -> str:
        now = self.clock()
        local = now.astimezone(EASTERN)
        day = local.date()
        # Never into quiet hours: it must be SAID by five minutes before they start.
        cutoff = _at(day, self.quiet_start_hour) - timedelta(minutes=5)
        if not (_at(day, self.start_hour) <= now < cutoff):
            return "outside the window"
        if self.done == day:
            return "done"
        key = f"{EVENING_SOURCE}:{day.isoformat()}"
        store = self.desk.store
        if store.recently_sent(key, _at(day, 0)) or store.decided(key, _at(day, 0)):
            self.done = day
            return "done"
        if EVENING_TOPIC in {str(m).strip().lower() for m in store.prefs(self.desk.budget_default).muted}:
            return "muted"
        try:
            reading = self.read()
        except Exception:
            log.warning("evening: can't read the house this pass", exc_info=True)
            return "no reading"
        if reading.room not in self.rooms or not reading.confirmed:
            return "not in a room for it"
        held_by = self.desk.voice_held(now)
        if held_by:
            return f"held: {held_by}"
        try:
            facts = self.gather(now, [], [], day=day + timedelta(days=1))
        except Exception:
            log.warning("evening: gathering tomorrow failed; no heads-up tonight", exc_info=True)
            self.done = day
            return "no facts"
        if not worth_saying(facts, early_hour=self.early_hour):
            if not facts.complete:                # a source didn't answer: a gap, not a quiet day
                return "incomplete reading"
            # A quiet tomorrow: nothing said, and recorded so a restart doesn't ask again.
            try:
                store.add(ReachEvent(source=EVENING_SOURCE, topic=EVENING_TOPIC, headline="(a quiet tomorrow)",
                                     brief=True, dedupe_key=key), now, status="dropped",
                          reason="nothing worth a heads-up", not_before=None)
            except Exception:
                log.warning("evening: couldn't record the quiet evening; this process remembers it", exc_info=True)
            self.done = day
            log.info("evening %s: nothing worth a heads-up", day)
            return "nothing worth saying"
        text = ""
        try:
            text = (self.compose(facts) or "").strip()
        except Exception:
            log.warning("evening: her words failed; the plain version goes", exc_info=True)
        if not text:
            text = plain_evening(facts)
        outcome = self.desk.reach_out(ReachEvent(
            source=EVENING_SOURCE, topic=EVENING_TOPIC, headline=text, room=reading.room, brief=True,
            dedupe_key=key, expires_at=cutoff))
        self.done = day
        log.info("evening %s: %s", day, outcome)
        return outcome

    def run_forever(self) -> None:  # pragma: no cover - thin loop over tick()
        log.info("evening heads-up armed | %s, %02d:00 until quiet hours", "/".join(self.rooms), self.start_hour)
        _loop(self.tick, "evening")


def start_evening_heads_up(heads_up: "EveningHeadsUp | None") -> threading.Thread | None:
    if heads_up is None:
        return None
    thread = threading.Thread(target=heads_up.run_forever, daemon=True, name="evening-heads-up")
    thread.start()
    return thread


# --- production legs (wired from settings by morning_brief_for) --------------------------


def _csv(value) -> list[str]:
    return [e.strip() for e in str(value or "").split(",") if e.strip()]


def _ha(settings):
    base = str(getattr(settings, "ha_base_url", "")).rstrip("/")
    return base, {"Authorization": f"Bearer {settings.ha_token.get_secret_value()}"}


def reader_for(settings, *, post=None) -> Callable[[], Reading]:
    """One HA template call: the settled room, whether motion backs it, and the latest wake
    signal (an automation's last run, or a sensor that's on and came on). post(url, headers,
    json) is the test seam."""
    import json

    import httpx

    settled = str(getattr(settings, "presence_settled_entity", "") or "").strip()
    wakes = _csv(getattr(settings, "brief_wake_entities", ""))
    signals = ", ".join(
        f"state_attr({e!r}, 'last_triggered')" if e.startswith("automation.")
        else f"(states[{e!r}].last_changed if is_state({e!r}, 'on') else none)"
        for e in wakes)
    template = ("{{ {'room': states(" + repr(settled) + "), 'confidence': state_attr(" + repr(settled)
                + ", 'confidence'), 'wakes': [" + signals + "] | reject('none') | map('string') | list} | to_json }}")
    base, headers = _ha(settings)
    send = post or (lambda url, hdrs, body: httpx.post(url, headers=hdrs, json=body, timeout=5.0))

    def read() -> Reading:
        r = send(f"{base}/api/template", headers, {"template": template})
        r.raise_for_status()
        data = json.loads(r.text)
        times = []
        for raw in data.get("wakes") or []:
            try:
                t = datetime.fromisoformat(str(raw))
            except ValueError:
                continue
            times.append(t if t.tzinfo else t.replace(tzinfo=timezone.utc))
        return Reading(room=str(data.get("room") or "").strip().lower(),
                       confirmed=str(data.get("confidence") or "") == "confirmed",
                       woke_at=max(times) if times else None)

    return read


#: Conditions worth a heads-up the evening before.
WET = {"rainy", "pouring", "lightning", "lightning-rainy", "hail", "snowy", "snowy-rainy", "exceptional"}

#: HA's condition codes, as a person would say them.
CONDITIONS = {"partlycloudy": "partly cloudy", "clear-night": "clear", "lightning": "thunderstorms",
              "lightning-rainy": "thunderstorms", "pouring": "heavy rain", "rainy": "rain",
              "snowy-rainy": "a wintry mix", "windy-variant": "windy", "exceptional": "severe weather"}


def _weather_line(forecast: dict) -> str:
    raw = str(forecast.get("condition") or "")
    cond = CONDITIONS.get(raw, raw.replace("-", " "))
    bits = [cond] if cond else []
    if forecast.get("temperature") is not None:
        bits.append(f"high {round(float(forecast['temperature']))}")
    if forecast.get("templow") is not None:
        bits.append(f"low {round(float(forecast['templow']))}")
    if forecast.get("precipitation_probability") is not None:
        bits.append(f"{round(float(forecast['precipitation_probability']))}% chance of rain")
    return ", ".join(bits)


def gatherer_for(settings, *, get=None, post=None) -> Callable[[datetime, list, list], Facts]:
    """The day's facts from HA: the daily forecast, and what's left on today's calendars.
    Work calendars (owner_meeting_calendars) give times only; personal ones (brief_calendars)
    give titles too. Each source fails open (logged) and the brief goes without it."""
    import httpx
    from urllib.parse import urlencode

    from aerys_v2.tools.calendar_events import _fmt_time, _parse_when

    base, headers = _ha(settings)
    weather_entity = str(getattr(settings, "brief_weather_entity", "") or "").strip()
    work = _csv(getattr(settings, "owner_meeting_calendars", ""))
    personal = [e for e in _csv(getattr(settings, "brief_calendars", "")) if e not in work]
    fetch = get or (lambda url, hdrs: httpx.get(url, headers=hdrs, timeout=5.0))
    send = post or (lambda url, hdrs, body: httpx.post(url, headers=hdrs, json=body, timeout=10.0))

    def forecast_for(day: date, today: date) -> dict:
        """The daily forecast entry for `day` (entries are dated in UTC; matched in local time)."""
        if not weather_entity:
            return {}
        r = send(f"{base}/api/services/weather/get_forecasts?return_response", headers,
                 {"entity_id": weather_entity, "type": "daily"})
        r.raise_for_status()
        forecast = ((r.json().get("service_response") or {}).get(weather_entity) or {}).get("forecast") or []
        for entry in forecast:
            try:
                if datetime.fromisoformat(str(entry.get("datetime"))).astimezone(EASTERN).date() == day:
                    return entry
            except (TypeError, ValueError):
                continue
        # Undated entries only: the first is today's (HA lists from today on). A dated list that
        # lacks the day is no answer for it; tomorrow always needs a date.
        undated = forecast and not any(e.get("datetime") for e in forecast)
        return forecast[0] if undated and day == today else {}

    def events(start_at: datetime, day: date, failed: list) -> list[tuple[datetime, str, bool]]:
        """(start, line, personal) for what's on `day` from start_at on, in time order. A
        calendar that can't be read is skipped and noted in `failed`."""
        end = datetime.combine(day + timedelta(days=1), time(0), tzinfo=EASTERN)
        window = urlencode({"start": start_at.isoformat(), "end": end.astimezone(timezone.utc).isoformat()})
        out = []
        for entity in [*work, *personal]:
            mine = []
            try:          # the read, the shape and every event: one calendar's failure boundary
                r = fetch(f"{base}/api/calendars/{entity}?{window}", headers)
                r.raise_for_status()
                items = r.json()
                if not isinstance(items, list):
                    raise ValueError("not a list of events")
                for ev in items:
                    if not isinstance(ev, dict):
                        raise ValueError("not an event")
                    start, start_day = _parse_when(ev.get("start"))
                    end_at, _ = _parse_when(ev.get("end"))
                    title = "a work meeting" if entity in work else (str(ev.get("summary") or "").strip() or "something")
                    if start is None:
                        if start_day is not None and entity not in work:      # all-day: personal only
                            mine.append((datetime.combine(start_day, time(0), tzinfo=EASTERN), f"all day: {title}", True))
                        continue
                    start = start if start.tzinfo else start.replace(tzinfo=EASTERN)
                    when = _fmt_time(start.astimezone(EASTERN))
                    if end_at is not None:
                        end_at = end_at if end_at.tzinfo else end_at.replace(tzinfo=EASTERN)
                        when += f" to {_fmt_time(end_at.astimezone(EASTERN))}"
                    mine.append((start, f"{when}: {title}", entity not in work))
            except Exception as e:
                log.warning("brief: a calendar read failed (%s); going without it", type(e).__name__)
                failed.append(entity)
                continue
            out += mine
        return sorted(out, key=lambda x: x[0])

    def gather(now: datetime, held: list, quiet: list, day: date | None = None) -> Facts:
        """Today from now on, or (day given) all of that day: the evening heads-up's tomorrow."""
        target = day or now.astimezone(EASTERN).date()
        start_at = now if day is None else datetime.combine(day, time(0), tzinfo=EASTERN)
        failed: list = []
        weather, wet = "", False
        try:                      # the read AND the reading: a malformed forecast costs only the weather
            entry = forecast_for(target, now.astimezone(EASTERN).date())
            if entry:
                weather = _weather_line(entry)
                rain = entry.get("precipitation_probability")
                wet = str(entry.get("condition") or "") in WET or (rain is not None and float(rain) >= 40)
            elif weather_entity:
                failed.append(weather_entity)          # no entry for the day is a gap, not a quiet sky
        except Exception as e:
            log.warning("brief: the forecast read failed (%s); going without it", type(e).__name__)
            failed.append(weather_entity)
        found = events(start_at, target, failed)
        timed = [t for t, line, _ in found if not line.startswith("all day")]
        return Facts(weather=weather, events=[line for _, line, _ in found], held=held, quiet=quiet,
                     first_at=min(timed) if timed else None, personal=sum(1 for *_, mine in found if mine),
                     wet=wet, complete=not failed)

    return gather


def composer_for(settings, *, model=None, prompt: str | None = None,
                 text_of: Callable[[Facts], str] | None = None) -> Callable[[Facts], str]:
    from langchain_core.messages import HumanMessage, SystemMessage

    def compose(facts: Facts) -> str:
        from aerys_v2.factory import build_model, load_soul
        from aerys_v2.service import _reply_text

        soul = load_soul(settings.soul_file_path)
        reply = (model or build_model(settings)).invoke([
            SystemMessage(content=soul + (prompt or BRIEF_PROMPT)),
            HumanMessage(content=(text_of or facts_text)(facts)),
        ])
        return _reply_text(reply).strip()

    return compose


def evening_heads_up_for(settings, desk) -> EveningHeadsUp | None:
    """None unless armed: evening_heads_up on, a desk, HA, and a speaker in one of its rooms."""
    if not getattr(settings, "evening_heads_up", False) or desk is None or getattr(settings, "ha_token", None) is None:
        return None
    from aerys_v2.kael_line import parse_room_devices

    speakers = parse_room_devices(getattr(settings, "reach_room_devices", ""))
    rooms = tuple(r for r in (x.strip().lower() for x in _csv(getattr(settings, "evening_rooms", "office,kitchen")))
                  if r in speakers and r != "bedroom")
    if not rooms:
        log.warning("evening heads-up: no speaker in any of its rooms; not armed")
        return None
    return EveningHeadsUp(
        desk, read=reader_for(settings), gather=gatherer_for(settings),
        compose=composer_for(settings, prompt=EVENING_PROMPT, text_of=evening_facts_text), rooms=rooms,
        start_hour=int(getattr(settings, "evening_start_hour", 19)),
        quiet_start_hour=int(getattr(settings, "owner_quiet_start_hour", 21)))


def morning_brief_for(settings, desk, *, replied_since=None) -> MorningBrief | None:
    """None unless armed: morning_brief on, a desk, HA, and a speaker in the brief's room."""
    if not getattr(settings, "morning_brief", False) or desk is None or getattr(settings, "ha_token", None) is None:
        return None
    from aerys_v2.kael_line import parse_room_devices

    room = str(getattr(settings, "brief_room", "office") or "office").strip().lower()
    if room not in parse_room_devices(getattr(settings, "reach_room_devices", "")):
        log.warning("morning brief: no speaker configured for the %s; not armed", room)
        return None
    return MorningBrief(
        desk, read=reader_for(settings), gather=gatherer_for(settings), compose=composer_for(settings),
        replied_since=replied_since or getattr(desk, "replied_since", None), room=room,
        start_hour=int(getattr(settings, "brief_start_hour", 5)),
        solo_hour=int(getattr(settings, "brief_solo_hour", 8)),
        cutoff_hour=int(getattr(settings, "brief_cutoff_hour", 13)))
