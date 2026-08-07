"""calendar_events — what's coming up, read straight off the house calendars.

Gap #27, self-filed by Aerys (2026-08-05) and owner-blessed 2026-08-07: "I
don't have a way to pull a full list of events off your calendar directly —
the tool only tells me whether something's happening *right now*, not what's
scheduled ahead." home_control's get_state on a calendar.* entity really does
only expose the current/next event; the actual schedule lives behind Home
Assistant's calendar API (GET /api/calendars/<entity>?start&end).

ALLOWLIST, NOT AUTO-DISCOVERY (deliberate): the hub aggregates every calendar
any integration syncs — including other people's shared calendars. Reading
THOSE into a reply would leak schedules the owner never asked her to know.
So the tool reads exactly the entities named in HA_CALENDAR_ENTITIES and
nothing else; adding a calendar is an env edit, not a code change.

Read-only end to end (calendar GETs can't mutate), rides the HOME half of the
action stack so the existing action allowlist gates who can ask. Transient
blips heal with the same single-retry used by home_control reads.

Failure posture: ToolNode contract — honest strings, never raises.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx

from aerys_v2.tools.home_control import _get_with_retry

log = logging.getLogger(__name__)

DAYS_DEFAULT = 7
DAYS_MAX = 31
EVENT_CAP = 40  # keep the tool message inside budget; truncation is announced


def calendar_set(csv: str) -> tuple[str, ...]:
    """Parse HA_CALENDAR_ENTITIES ('' -> empty tuple, order preserved)."""
    return tuple(e.strip() for e in csv.split(",") if e.strip())


def _label(entity_id: str) -> str:
    """A short human label from the entity id ('calendar.home' -> 'home')."""
    return entity_id.split(".", 1)[-1].replace("_", " ")


def _parse_when(raw: dict | None) -> tuple[datetime | None, date | None]:
    """HA event start/end: {'dateTime': iso} for timed, {'date': iso} all-day."""
    if not isinstance(raw, dict):
        return None, None
    if raw.get("dateTime"):
        try:
            return datetime.fromisoformat(raw["dateTime"]), None
        except ValueError:
            return None, None
    if raw.get("date"):
        try:
            return None, date.fromisoformat(raw["date"])
        except ValueError:
            return None, None
    return None, None


def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def _fmt_day(d: date) -> str:
    return d.strftime("%a %b ") + str(d.day)


def build_calendar_tool(
    *,
    base_url: str,
    token: str,
    entities: tuple[str, ...],
    client: httpx.Client | None = None,
):
    """Close over config and return the calendar_events tool (test seam: inject
    an httpx.Client on a MockTransport)."""
    from langchain_core.tools import tool

    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    http = client or httpx.Client(timeout=10.0)

    @tool
    def calendar_events(days: int = DAYS_DEFAULT) -> str:
        """List the owner's upcoming calendar events.

        CALL THIS TOOL whenever the user asks what's on their calendar or
        schedule — "what's on my calendar", "do I have anything tomorrow /
        this week", "when is my next appointment", "what's coming up".
        Never answer schedule questions from memory alone; memories record
        what was SAID about events, this reads the calendar itself.

        days: how many days ahead to look (default 7, max 31). Use a bigger
        window when the user asks about something further out.

        Returns events in order, one per line, grouped under day headers,
        each tagged with the calendar it came from. Times are the owner's
        local timezone. An empty window says so honestly.
        """
        try:
            span = int(days)
        except (TypeError, ValueError):
            span = DAYS_DEFAULT
        span = max(1, min(span, DAYS_MAX))

        now = datetime.now().astimezone()
        # urlencode is load-bearing: a UTC box renders isoformat() with +00:00,
        # and a RAW '+' in a query string decodes as a SPACE → HA 400s the
        # window (found live on first deploy — the container clock is UTC).
        window = urlencode(
            {"start": now.isoformat(), "end": (now + timedelta(days=span)).isoformat()}
        )

        events: list[tuple[Any, str, str]] = []  # (sort_key, day_key, line)
        unreachable: list[str] = []
        for entity in entities:
            try:
                r = _get_with_retry(
                    http,
                    f"{base}/api/calendars/{entity}?{window}",
                    headers,
                )
                r.raise_for_status()
                items = r.json()
            except (httpx.HTTPError, ValueError) as e:
                log.warning("calendar read failed for %s: %s", entity, e)
                unreachable.append(_label(entity))
                continue
            if not isinstance(items, list):
                unreachable.append(_label(entity))
                continue
            for ev in items:
                if not isinstance(ev, dict):
                    continue
                summary = str(ev.get("summary") or "(untitled)").strip()
                s_dt, s_day = _parse_when(ev.get("start"))
                e_dt, e_day = _parse_when(ev.get("end"))
                if s_dt is not None:
                    when = f"{_fmt_time(s_dt)}–{_fmt_time(e_dt)}" if e_dt else _fmt_time(s_dt)
                    day = s_dt.date()
                    sort_key = (day.isoformat(), 1, s_dt.isoformat())
                elif s_day is not None:
                    # HA all-day ends are EXCLUSIVE — a one-day event ends tomorrow.
                    last = e_day - timedelta(days=1) if e_day else s_day
                    when = "all day" if last <= s_day else f"all day through {_fmt_day(last)}"
                    day = s_day
                    sort_key = (day.isoformat(), 0, summary)
                else:
                    continue
                line = f"  {when} — {summary} [{_label(entity)}]"
                loc = str(ev.get("location") or "").strip()
                if loc:
                    line += f" @ {loc}"
                events.append((sort_key, _fmt_day(day), line))

        if unreachable and not events:
            return (
                "The calendar is unreachable right now "
                f"(couldn't read: {', '.join(unreachable)})."
            )
        if not events:
            horizon = "day" if span == 1 else f"{span} days"
            return f"Nothing on the calendar for the next {horizon}."

        events.sort(key=lambda t: t[0])
        truncated = len(events) > EVENT_CAP
        events = events[:EVENT_CAP]

        lines: list[str] = []
        current_day = None
        for _, day_hdr, line in events:
            if day_hdr != current_day:
                lines.append(f"{day_hdr}:")
                current_day = day_hdr
            lines.append(line)
        if truncated:
            lines.append(f"[showing the first {EVENT_CAP} events — the window has more]")
        if unreachable:
            lines.append(f"[couldn't read: {', '.join(unreachable)} — answers may be partial]")
        return "\n".join(lines)

    return calendar_events
