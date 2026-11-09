"""kael_tickets — what she has asked Kael, in her hands (board #176).

Every message_kael opens a ticket (aerys_v2/tickets.py). This tool is the rest of the
verbs: check one, list them, add to one, cancel one. message_kael is "start". The verbs
match the async-task verbs on purpose, so Kael could later sit behind the same tools as
her home body without a rebuild.

Owner-side only, like message_kael (wired beside it in factory.py). Failure posture:
ToolNode contract — honest strings, never raises.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from aerys_v2.tickets import OPEN, OUTBOX_LIMIT, UPDATE_LIMIT, label, parse_id

log = logging.getLogger(__name__)

_STATE = {
    "submitted": "sent, waiting on Kael",
    "working": "Kael is on it",
    "input-required": "Kael asked something back",
    "completed": "answered",
    "failed": "Kael couldn't do it",
    "canceled": "canceled",
}


def _short(text: str | None, n: int = 160) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def _line(t) -> str:
    state = _STATE.get(t.status, t.status)
    if t.status in OPEN and t.outbox:
        state += " (not through to him yet; retrying)"
    return f"- {label(t.id)}: {state}. Asked: \"{_short(t.ask, 100)}\""


def build_kael_tickets_tool(store, send, *, clock=None):
    """store: a TicketStore. send(ticket, text) -> bool: her line, tickets.line_sender."""
    from langchain_core.tools import tool

    now = clock or (lambda: datetime.now(timezone.utc))

    @tool
    def kael_tickets(action: str, ticket: str = "", message: str = "") -> str:
        """Check on, add to, cancel or list what you've asked Kael. Every message_kael
        opens a ticket (T-12); this is how you follow one up.

        CALL THIS when the owner asks whether Kael has answered, what Kael said, or what
        you're waiting on him for; when you need to add something to an ask you already
        sent (or answer a question Kael asked back); or when an ask isn't needed anymore.
        Never re-send the same ask with message_kael: add to its ticket here.

        action: "check" (one ticket: its state and Kael's answer), "list" (recent
          tickets), "update" (send more on an open ticket), or "cancel".
        ticket: the ticket id, e.g. "T-12" (required for check, update and cancel).
        message: what to add (required for update).

        Returns what's true right now — never claim Kael answered or got something
        unless this says so.
        """
        act = (action or "").strip().lower()
        try:
            if act == "list":
                items = store.recent(limit=8)
                if not items:
                    return "You haven't asked Kael anything yet."
                return "Your tickets with Kael, newest first:\n" + "\n".join(_line(t) for t in items)
            if act not in ("check", "update", "cancel"):
                return 'NOT DONE: action must be "check", "list", "update" or "cancel".'
            tid = parse_id(ticket)
            if tid is None:
                return f'NOT DONE: {act} needs the ticket id (like "T-12"); call with action "list" to see them.'
            t = store.get(tid)
            if t is None:
                return f"NOT DONE: there is no ticket {label(tid)}; call with action \"list\" to see them."
            if act == "check":
                out = _line(t)
                if t.answer:
                    out += f"\nKael's answer: {t.answer}"
                # His latest note shows whatever the state: a correction he sends after an answer
                # (as 'working') must still be readable once the ticket is closed (11/09).
                if t.progress and t.progress != t.answer:
                    out += f"\nKael's latest note: {t.progress}"
                return out
            if act == "cancel":
                if t.status not in OPEN:
                    return f"{label(tid)} is already closed ({_STATE.get(t.status, t.status)}); nothing to cancel."
                kael_has_it = t.delivered_at is not None   # he got at least the ask itself
                store.cancel(tid, now())
                if kael_has_it:      # tell him it's withdrawn (best effort)
                    send(t, "canceled: no longer needed, nothing more to do on this one.")
                return f"Canceled {label(tid)}."
            words = " ".join((message or "").split())
            if not words:
                return "NOT DONE: update needs the message to add."
            if t.status not in OPEN:
                return (f"NOT DONE: {label(tid)} is closed ({_STATE.get(t.status, t.status)}). "
                        "For something new, send it with message_kael.")
            if len(words) > UPDATE_LIMIT:
                return f"NOT DONE: keep an update under {UPDATE_LIMIT:,} characters; split it or send the gist."
            text = f"update: {words}"
            if len(t.outbox or "") + 1 + len(text) > OUTBOX_LIMIT:
                return (f"NOT DONE: {label(tid)} already has a lot waiting to reach Kael (his line "
                        "hasn't taken it yet). Wait until it goes through, or keep this shorter.")
            t, accepted = store.from_her(tid, text, now())
            if not accepted:
                return f"NOT DONE: {label(tid)} couldn't take more right now; nothing was added."
            sent = t.outbox
            if send(t, sent):
                store.delivered(tid, now(), sent)
                return f"Added to {label(tid)}; Kael has it now."
            return (f"Added to {label(tid)}, but Kael's line didn't take it right now; it's "
                    "retried automatically. Don't send it again.")
        except Exception:
            log.warning("kael_tickets %s failed", act, exc_info=True)
            return "NOT DONE: the ticket ledger couldn't be reached; nothing changed."

    return kael_tickets
