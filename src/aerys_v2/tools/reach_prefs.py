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

from aerys_v2.reach import BUDGET_MAX, BUDGET_MIN, Prefs

log = logging.getLogger(__name__)

#: Topics the desk knows. Safety warnings and answers he asked for can't be muted.
TOPICS = ("kael", "weather", "calendar", "email", "follow-ups")


def build_reach_prefs_tool(store, *, default_budget: int = 3, clock=None):
    from langchain_core.tools import tool

    now = clock or (lambda: datetime.now(timezone.utc))

    @tool
    def reach_prefs(action: str, topic: str = "", his_words: str = "") -> str:
        """Change how often you reach out to the owner on your own, or what about, because he
        told you to. CALL THIS when he asks for less or more, in whatever words he uses:
        "ease up", "chill out", "you're pinging me a lot" mean less; "keep me posted more",
        "annoy me more", "check in more" mean more. And when he asks to stop or restart
        hearing about one kind of thing: "stop telling me about the weather" (mute weather),
        "you can tell me about email again" (unmute email). Also when he asks what you'll
        bring him on your own (show).

        action: "less", "more", "mute", "unmute" or "show".
        topic: for mute and unmute: kael (news from Kael), weather, calendar, email or
          follow-ups. Storm warnings and answers he asked for are never muted.
        his_words: what he said, briefly, for the record.

        Returns what changed, or what's set now. Tell him plainly what you changed.
        """
        act = (action or "").strip().lower()
        words = " ".join((his_words or "").split())[:300]
        try:
            prefs = store.prefs(default_budget)
            if act == "show":
                muted = ", ".join(prefs.muted) if prefs.muted else "nothing"
                return (f"Up to {prefs.budget} unprompted message(s) a day, plus the briefs and anything "
                        f"he asked for. Muted: {muted}.")
            if act in ("less", "more"):
                budget = max(BUDGET_MIN, min(BUDGET_MAX, prefs.budget + (1 if act == "more" else -1)))
                if budget == prefs.budget:
                    edge = "the most" if act == "more" else "the least"
                    return f"Already at {edge} (up to {budget} a day); nothing changed."
                store.set_prefs(Prefs(budget=budget, muted=prefs.muted), now(), f"budget {prefs.budget} -> {budget}", words)
                return f"Changed: up to {budget} unprompted message(s) a day now (was {prefs.budget})."
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
                store.set_prefs(Prefs(budget=prefs.budget, muted=muted), now(), f"{act} {key}", words)
                return (f"Changed: you won't bring up {key} on your own now." if act == "mute"
                        else f"Changed: you can bring up {key} on your own again.")
            return 'NOT DONE: action must be "less", "more", "mute", "unmute" or "show".'
        except Exception:
            log.warning("reach_prefs %s failed", act, exc_info=True)
            return "NOT DONE: the reach settings couldn't be reached; nothing changed."

    return reach_prefs
