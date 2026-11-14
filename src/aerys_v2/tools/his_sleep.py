"""his_sleep — his sleep, as his phone reports it to Home Assistant.

The owner asked (11/13) for her to have "the same full view of me" Kael has; Kael reads these sensors
to look out for him, and this gives her the same read. Read-only, and only to him in a room that's
just him: his typed line in a DM, or a voice she's sure is his, like his mail (an unsure voice may be
a guest; a guild, a group or a sealed room has other readers). A live reading, never a stored fact:
last night's numbers are wrong by tonight (the thin-memory lesson).

HA_SLEEP_SENSORS names the sensors (night / last / now); see config.py. Sensor timestamps without a
zone are read as UTC (the phone's are "Z"); old readings carry their date, never "last night".
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

import httpx
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

log = logging.getLogger(__name__)

UNSURE = ("Couldn't read his sleep: I share that only when I'm sure it's you. Type it to me, or ask "
          "again when I know your voice.")
NOT_HIM = "Couldn't read his sleep: only the owner can ask about his sleep."
SHARED = "Couldn't read his sleep here: others can read this room. Ask me in a DM, or by voice."
_DEAD = frozenset({"unknown", "unavailable", "none", ""})
#: a now-reading older than this is "last reading", not "right now"
NOW_FRESH = timedelta(hours=2)


def parse_sensors(csv: str) -> dict[str, str]:
    """'night=sensor.a,last=sensor.b,now=sensor.c' -> {'night': 'sensor.a', ...} (known keys only)."""
    out: dict[str, str] = {}
    for part in (csv or "").split(","):
        key, sep, entity = part.strip().partition("=")
        if sep and key.strip() in ("night", "last", "now") and entity.strip():
            out[key.strip()] = entity.strip()
    return out


def _number(value, low: float, high: float) -> float:
    n = float(value)
    if not math.isfinite(n) or not low <= n <= high:
        raise ValueError(f"{value!r} is out of range")
    return n


#: a duration sensor's unit -> minutes per unit; any other unit is refused, never guessed (Codex 11/14)
_PER_MINUTE = {"min": 1.0, "h": 60.0, "s": 1 / 60, "ms": 1 / 60000}


def _minutes(value, unit=None) -> str:
    unit = str(unit or "min").strip()
    if unit not in _PER_MINUTE:
        raise ValueError(f"unit {unit!r}")
    m = int(round(_number(float(value) * _PER_MINUTE[unit], 0, 48 * 60)))
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60} min"


def _moment(raw) -> datetime:
    """An ISO time or epoch milliseconds, as an aware UTC datetime (a zoneless ISO time is UTC)."""
    if isinstance(raw, (int, float)) or str(raw).isdigit():
        return datetime.fromtimestamp(int(raw) / 1000, tz=timezone.utc)
    dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _when(moment: datetime, now: datetime) -> str:
    """In his clock: '2:26 AM' today, 'Fri 11:04 PM' this past week, 'Jan 3, 11:04 PM' before that."""
    from aerys_v2.hands_gate import EASTERN

    local, today = moment.astimezone(EASTERN), now.astimezone(EASTERN).date()
    clock = local.strftime("%I:%M %p").lstrip("0")
    if local.date() == today:
        return clock
    if timedelta(0) < today - local.date() < timedelta(days=6):
        return f"{local.strftime('%a')} {clock}"
    return f"{local.strftime('%b')} {local.day}, {clock}"


def _night(s: dict, now: datetime) -> str | None:
    a = s.get("attributes") or {}
    total = _minutes(s["state"], a.get("unit_of_measurement"))
    end = _moment(a["last_end"]) if a.get("last_end") else None
    stretches = (f", in {int(a['sessions'])} stretches"
                 if str(a.get("sessions") or "").isdigit() and int(a["sessions"]) > 1 else "")
    if end is None:
        return f"Latest night his phone reported (no date given): {total} asleep{stretches}."
    from aerys_v2.hands_gate import EASTERN

    if end.astimezone(EASTERN).date() != now.astimezone(EASTERN).date():
        # "last night" only for a night that ended today; a missed sync says its date (Codex 11/14)
        return f"Latest night his phone reported (ended {_when(end, now)}): {total} asleep{stretches}."
    return f"Last night: {total} asleep{stretches}; the last one ended {_when(end, now)}."


def _last(s: dict, now: datetime) -> str | None:
    a = s.get("attributes") or {}
    if not (a.get("startTime") and a.get("endTime")):
        return None
    return (f"Latest stretch: {_when(_moment(a['startTime']), now)} to {_when(_moment(a['endTime']), now)} "
            f"({_minutes(s['state'], a.get('unit_of_measurement'))}).")


def _now(s: dict, now: datetime) -> str | None:
    a = s.get("attributes") or {}
    if str(a.get("unit_of_measurement") or "%").strip() != "%":
        raise ValueError(f"unit {a.get('unit_of_measurement')!r}")
    pct = int(round(_number(s["state"], 0, 100)))
    stamp = a.get("timestamp") or s.get("last_changed")
    if not stamp:
        return f"His phone's latest asleep reading: {pct}% (no time given)."
    at = _moment(stamp)
    if now - at > NOW_FRESH:
        return f"Last reading ({_when(at, now)}): his phone was {pct}% sure he was asleep."
    return f"Right now: his phone is {pct}% sure he's asleep (as of {_when(at, now)})."


def describe(states: dict[str, dict], now: datetime) -> str:
    """His sleep in plain lines, from whichever sensors answered; each line stands or falls alone."""
    lines = []
    for key, render in (("night", _night), ("last", _last), ("now", _now)):
        s = states.get(key)
        if not s or str(s.get("state")).lower() in _DEAD:
            continue
        try:
            line = render(s, now)
        except Exception:
            log.info("his_sleep: an odd %s reading, left out", key, exc_info=True)
            continue
        if line:
            lines.append(line)
    if not lines:
        return "His phone hasn't reported any sleep to Home Assistant that I can read, so I can't tell."
    return ("His sleep, from his phone (it syncs late, so the newest stretch can be missing):\n- "
            + "\n- ".join(lines))


def build_his_sleep_tool(settings, *, client: httpx.Client | None = None, clock=None):
    from aerys_v2.hands_gate import OWNER_TEXT, OWNER_VOICE, asker

    sensors = parse_sensors(settings.ha_sleep_sensors)
    owner = str(settings.owner_person_id or "")
    base = settings.ha_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.ha_token.get_secret_value()}"}
    http = client or httpx.Client(timeout=5.0)
    clock = clock or (lambda: datetime.now(timezone.utc))

    def verdict(runtime) -> str | None:
        try:
            conf = runtime.config.get("configurable") or {}
            identity = conf.get("identity") or {}
            thread = str(conf.get("thread_id") or "")
            if not owner or str(identity.get("user_id") or "") != owner or thread.startswith("kael:"):
                return NOT_HIM
            if (identity.get("privacy_context") != "private" or identity.get("sealed_room")
                    or str(identity.get("channel_kind") or "dm") != "dm"):
                return SHARED                   # his own words in a room others read (Codex 11/14)
            return None if asker(identity, thread, owner) in (OWNER_TEXT, OWNER_VOICE) else UNSURE
        except Exception:
            return NOT_HIM

    @tool
    def his_sleep(runtime: ToolRuntime) -> str:
        """HIS sleep, as his phone reports it: last night's total and stretches, the latest stretch, and
        whether his phone thinks he's asleep right now. Owner only, in a private room. Use it when he
        asks how he slept, or when it would change how you look after him (a short or broken night: go
        gentler, keep it brief). Don't bring his sleep up unasked just to report it, and never nag. A
        live reading: never save it as a memory about him.
        """
        refusal = verdict(runtime)
        if refusal is not None:
            return refusal
        states: dict[str, dict] = {}
        for key, entity in sensors.items():
            try:
                r = http.get(f"{base}/api/states/{entity}", headers=headers)
                r.raise_for_status()
                body = r.json()
                if isinstance(body, dict):
                    states[key] = body
            except Exception:
                log.debug("his_sleep: %s didn't answer", entity, exc_info=True)
        try:
            return describe(states, clock())
        except Exception:
            log.warning("his_sleep: an odd sensor reading", exc_info=True)
            return "His phone's sleep numbers came back in a shape I can't read right now."

    return his_sleep
