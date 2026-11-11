"""When each line in her thread was said (ticket T-6, 2026-11-11).

The owner said on Monday "i have a doctor's appointment in venice tomorrow"; on Wednesday, by voice,
she read that line back out of the thread and told him the appointment was tomorrow. The thread's
messages carried no time, so "tomorrow" was read against today. Her long-term memories already carry
their dates (services/memory.py); conversation history did not.

The write half: every human line gets SAID_AT_KEY (UTC, ISO) in additional_kwargs as it's checkpointed.
The read half, with_said_dates: what a model is SHOWN of earlier days carries "[said Mon Nov 9, 2:39 PM]"
in front of the human line (Eastern time). Lines from earlier today stay as they are (the time line in
her prompt already says what today is), and the current turn is never labelled. A line from before
stamping existed that uses a relative day word ("tomorrow", "tonight") says its date wasn't recorded,
so she asks instead of guessing. Her own replies are never labelled: she copies patterns she sees in
her own lines (see factory._untag_own_replies); the prompt says a reply right after a dated line was said
then too. The checkpoint keeps the literal record; this is only what she is shown.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

SAID_AT_KEY = "said_at"
EASTERN = ZoneInfo("America/New_York")
UNRECORDED = "[said earlier; its date wasn't recorded]"
#: Words that only mean something against the day they were said.
RELATIVE = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|this (morning|afternoon|evening|week|weekend)|"
    r"next (week|weekend|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"last night|later today|in (a|an|two|three|\d+) (hour|hours|day|days))\b",
    re.IGNORECASE,
)


def said_at_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def said_at_of(message: object) -> datetime | None:
    raw = (getattr(message, "additional_kwargs", None) or {}).get(SAID_AT_KEY)
    if not raw:
        return None
    try:
        when = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def said_label(when: datetime) -> str:
    """"[said Mon Nov 9, 2:39 PM]" in Eastern time (portable format: no %-d / %-I)."""
    et = when.astimezone(EASTERN)
    hour12 = et.strftime("%I").lstrip("0") or "12"
    return f"[said {et:%a %b} {et.day}, {hour12}:{et:%M} {et:%p}]"


def with_said_dates(messages: list, now: datetime | None = None) -> list:
    """Copies of earlier days' human lines with when they were said in front; everything else as is."""
    today = (now or datetime.now(timezone.utc)).astimezone(EASTERN).date()
    last_human = max((i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"), default=-1)
    shown = []
    for i, m in enumerate(messages):
        content = getattr(m, "content", None)
        if getattr(m, "type", "") != "human" or i == last_human or not isinstance(content, str):
            shown.append(m)
            continue
        when = said_at_of(m)
        if when is not None:
            label = said_label(when) if when.astimezone(EASTERN).date() != today else ""
        else:
            label = UNRECORDED if RELATIVE.search(content) else ""
        shown.append(m.model_copy(update={"content": f"{label} {content}"}) if label else m)
    return shown
