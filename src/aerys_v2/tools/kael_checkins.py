"""kael_checkins — the check-ins that tap Kael on a schedule, in her hands.

Owner ask (2026-10-12): a recurring job lives with her (see aerys_v2/checkins.py),
and stopping it should be as easy as telling her. This tool lists them and turns one
off or back on. Adding one stays with the owner and Kael (a row in v2_kael_checkins):
a check-in starts work in Kael's session, so who can create them is kept narrow.

Owner-side only, like message_kael (wired beside it in factory.py). Failure posture:
ToolNode contract — honest strings, never raises.
"""

from __future__ import annotations

import logging

from aerys_v2.checkins import CheckinStore

log = logging.getLogger(__name__)


def _describe(item) -> str:
    state = "on" if item.active else "OFF"
    last = item.last_fired.isoformat() if item.last_fired else "never"
    return f"- {item.name}: {'/'.join(item.days)} at {item.at} (Eastern), {state}; last tapped Kael: {last}"


def build_kael_checkins_tool(store: CheckinStore):
    from langchain_core.tools import tool

    @tool
    def kael_checkins(action: str, name: str = "") -> str:
        """List, stop or resume the check-ins that tap Kael on a schedule.

        A check-in is a recurring job Kael runs for the household: at its days and
        time you send its message down your line to Kael, and Kael starts the work.
        CALL THIS when the owner asks what check-ins exist, or asks to stop, pause,
        cancel, resume or restart one ("stop the Thursday one", "turn the digest
        back on"). You cannot add or change one here; for that, the owner asks Kael.

        action: "list", "stop" or "resume".
        name: the check-in's name (from list), required for stop and resume.

        Returns the list, or what changed — never claim a change that didn't happen.
        """
        act = (action or "").strip().lower()
        try:
            if act == "list":
                items = store.list()
                if not items:
                    return "There are no check-ins."
                return "Check-ins:\n" + "\n".join(_describe(i) for i in items)
            if act in ("stop", "resume"):
                key = (name or "").strip()
                if not key:
                    return f"NOT DONE: {act} needs the check-in's name; call with action \"list\" first."
                if not store.set_active(key, act == "resume"):
                    return f"NOT DONE: there is no check-in named {key!r}; call with action \"list\" to see them."
                return (f"Stopped {key}: it won't tap Kael until it's resumed." if act == "stop"
                        else f"Resumed {key}: it taps Kael again at its next time.")
            return 'NOT DONE: action must be "list", "stop" or "resume".'
        except Exception:
            log.warning("kael_checkins %s failed", act, exc_info=True)
            return "NOT DONE: the check-ins store couldn't be reached; nothing changed."

    return kael_checkins
