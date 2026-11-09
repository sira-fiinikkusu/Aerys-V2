"""reach_prefs — how often she reaches out on her own, and about what, from his own words.

The owner, 2026-11-08 (D4): "3 a day plus briefs as a starting point but this can be
adaptive, i'd like for an option for me to say chill out or annoy me more (not exactly in
those words)." And D7: no emoji reactions, "this should be natural language". So he says it
however he says it, she understands it, and this tool turns it into a change the desk
enforces in code (reach.py): a daily budget (0 to 8) and topics he muted. Every change is
recorded with his words, so "why did she go quiet?" always has an answer.

Owner-side only, like kael_tickets (wired beside it in factory.py). Failure posture:
ToolNode contract — honest strings, never raises.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from datetime import timedelta

from dataclasses import replace

from aerys_v2.reach import BUDGET_MAX, BUDGET_MIN, CHECKINS_MAX, CHECKINS_MIN, NAP_DEFAULT, NAP_MAX

log = logging.getLogger(__name__)

#: Topics the desk knows. Safety warnings and answers he asked for can't be muted.
TOPICS = ("kael", "weather", "calendar", "email", "follow-ups", "appointments", "to-dos", "brief", "evening")


def build_reach_prefs_tool(store, *, default_budget: int = 3, clock=None):
    from langchain_core.tools import tool

    now = clock or (lambda: datetime.now(timezone.utc))

    @tool
    def reach_prefs(action: str, topic: str = "", his_words: str = "", minutes: int = 0) -> str:
        """Change how often you reach out to the owner on your own, or what about, because he
        told you to. CALL THIS when he asks for less or more, in whatever words he uses:
        "ease up", "chill out", "you're pinging me a lot" mean less; "keep me posted more",
        "annoy me more", "check in more" mean more. And when he asks to stop or restart
        hearing about one kind of thing: "stop telling me about the weather" (mute weather),
        "you can tell me about email again" (unmute email). Also when he asks what you'll
        bring him on your own (show). When he says he's taking a nap or lying down for a bit
        (nap): nothing you'd say aloud goes until he talks to you again or the time is up;
        when he says he's up (awake). Your unasked check-ins (about his appointments and things
        he said he'd do) have their own dial: "check on me more", "check in on me more often"
        (checkins_more); "check on me less", "you don't have to check up on me so much"
        (checkins_less); "stop checking up on me" (mute follow-ups); "don't ask about my
        appointments" (mute appointments); "don't chase my to-dos" (mute to-dos).

        action: "less", "more", "mute", "unmute", "show", "nap", "awake", "checkins_more" or
          "checkins_less".
        minutes: for nap, if he said how long (default 60, at most 180).
        topic: for mute and unmute: kael (news from Kael), weather, calendar, email,
          follow-ups (all your unasked check-ins), appointments, to-dos, brief (your spoken
          morning brief in his office), or evening (your heads-up about tomorrow). Storm
          warnings, answers he asked for and check-backs he said yes to are never muted.
        his_words: what he said, briefly, for the record.

        Returns what changed, or what's set now. Tell him plainly what you changed.
        """
        act = (action or "").strip().lower()
        words = " ".join((his_words or "").split())[:300]
        try:
            prefs = store.prefs(default_budget)
            if act == "show":
                muted = ", ".join(prefs.muted) if prefs.muted else "nothing"
                nap = (f" A nap hold runs until {prefs.hold_until:%H:%M} UTC or until he talks to you."
                       if prefs.hold_until is not None and prefs.hold_until > now() else "")
                return (f"Up to {prefs.budget} unprompted message(s) a day, plus the briefs and anything "
                        f"he asked for; up to {prefs.checkins} unasked check-in(s) of those. Muted: {muted}.{nap}")
            if act in ("less", "more"):
                budget = max(BUDGET_MIN, min(BUDGET_MAX, prefs.budget + (1 if act == "more" else -1)))
                if budget == prefs.budget:
                    edge = "the most" if act == "more" else "the least"
                    return f"Already at {edge} (up to {budget} a day); nothing changed."
                store.set_prefs(replace(prefs, budget=budget), now(), f"budget {prefs.budget} -> {budget}", words)
                return f"Changed: up to {budget} unprompted message(s) a day now (was {prefs.budget})."
            if act == "nap":
                span = timedelta(minutes=int(minutes)) if minutes and int(minutes) > 0 else NAP_DEFAULT
                span = min(span, NAP_MAX)
                start = now()
                store.set_prefs(replace(prefs, hold_since=start, hold_until=start + span),
                                start, f"nap {int(span.total_seconds() // 60)} min", words)
                return (f"Holding: nothing said aloud for up to {int(span.total_seconds() // 60)} minutes, or until "
                        "he talks to you again. Anything timely goes as a silent text.")
            if act == "awake":
                if prefs.hold_until is None:
                    return "There was no nap hold; nothing changed."
                store.set_prefs(replace(prefs, hold_since=None, hold_until=None), now(), "awake", words)
                return "The nap hold is off; you can speak to him again."
            if act in ("mute", "unmute"):
                key = (topic or "").strip().lower()
                if key not in TOPICS:
                    return f"NOT DONE: topic must be one of {', '.join(TOPICS)}."
                muted = [m for m in prefs.muted if m != key]
                if act == "mute":
                    if key in prefs.muted:
                        return f"{key} is already muted; nothing changed."
                    muted.append(key)
                elif key not in prefs.muted:
                    return f"{key} wasn't muted; nothing changed."
                store.set_prefs(replace(prefs, muted=muted), now(), f"{act} {key}", words)
                return (f"Changed: you won't bring up {key} on your own now." if act == "mute"
                        else f"Changed: you can bring up {key} on your own again.")
            if act in ("checkins_more", "checkins_less"):
                n = max(CHECKINS_MIN, min(CHECKINS_MAX, prefs.checkins + (1 if act == "checkins_more" else -1)))
                if n == prefs.checkins:
                    edge = "the most" if act == "checkins_more" else "none"
                    return f"Already at {edge} ({n} unasked check-ins a day); nothing changed."
                store.set_prefs(replace(prefs, checkins=n), now(), f"check-ins {prefs.checkins} -> {n}", words)
                return f"Changed: up to {n} unasked check-in(s) a day now (was {prefs.checkins})."
            return ('NOT DONE: action must be "less", "more", "mute", "unmute", "show", "nap", "awake", '
                    '"checkins_more" or "checkins_less".')
        except Exception:
            log.warning("reach_prefs %s failed", act, exc_info=True)
            return "NOT DONE: the reach settings couldn't be reached; nothing changed."

    return reach_prefs
