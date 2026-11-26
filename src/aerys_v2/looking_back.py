"""D99 (the owner, 11/25 12:46, "your pick works here"): in a group room, a household member's "looking back"
question goes to her tool side, where search_memory and search_history are.

D95-A put her search tools on the action graph. A group room's chat-shaped turn goes to the tool-less chat
side, so a household member's "remember when we talked about the router?" in the guild got the 5 recalled
lines and nothing more. This check lets ask() move only that kind of turn:
  - ONE Jev judgement: does the message ask her to remember or look back at something from before;
  - code decides: reroute at REROUTE_AT or above; anything else (a lower score, a timeout, an error, no key)
    leaves the turn on the chat side exactly as before (the old path, never a new failure);
  - who may be rerouted is decided in service.room_recall_eligible (household only, group rooms only), never
    here. Guests keep the recalled lines and no search tools.
Judgements from Jev, the decision in code (the TypeSafe rule). The state is the message alone.
"""
from __future__ import annotations

import contextvars
import math
import threading
import time

MODEL = "jev-1.14.0"
TIMEOUT_S = 1.0
REROUTE_AT = 0.7
MESSAGE_MAX = 600

QUESTION_ID = "looking_back"
QUESTIONS = {
    QUESTION_ID: {
        "type": "noul",
        "instructions": (
            "Does `message` ask Aerys to remember or look back at something from before: what someone said or "
            "did earlier, what happened, or when something was talked about?"
        ),
        "criteria": {
            "true": "He wants her to recall the past (\"remember when we talked about the router?\", "
                    "\"what did Sam say about Saturday?\", \"when did I mention the trip?\", "
                    "\"didn't you tell me the bag rules?\").",
            "false": "He asks about now or the future, asks her to do or look up something new, chats, or tells "
                     "her something (\"what's the weather tomorrow?\", \"turn on the lights\", \"look up the price "
                     "of a Switch 2\", \"lol nice\", \"I just got back from the store\").",
        },
    }
}


def build_state(message: str) -> dict:
    text = " ".join(str(message or "").split())
    return {"message": text if len(text) <= MESSAGE_MAX else text[:MESSAGE_MAX] + "…"}


def reroute(p: float | None) -> bool:
    """Send the turn to her tool side? Only on a real score at or above REROUTE_AT. None never reroutes."""
    return p is not None and p >= REROUTE_AT


class LookingBackCheck:
    """check(message) -> the chance the message asks her to look back, or None (unavailable: timeout, error,
    bad score, too many in flight). One request, bounded; a late answer is never used."""

    def __init__(self, client: object, *, model: str = MODEL, timeout_s: float = TIMEOUT_S):
        self.client, self.model, self.timeout_s = client, model, timeout_s
        self._inflight = threading.BoundedSemaphore(8)

    def __call__(self, message: str) -> float | None:
        started = time.monotonic()
        state = build_state(message)
        if not state["message"]:
            return None
        box: dict = {}
        done = threading.Event()

        def run():
            try:
                response = self.client.system_one(state=state, questions=QUESTIONS, model=self.model)
                v = response.answers[QUESTION_ID].noul
                if not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0:
                    box["p"] = float(v)
            except Exception:  # noqa: BLE001 -- the check never blocks a reply
                pass
            finally:
                box["at"] = time.monotonic()
                done.set()
                self._inflight.release()

        if not self._inflight.acquire(blocking=False):
            return None
        try:
            threading.Thread(target=contextvars.copy_context().run, args=(run,), daemon=True).start()
        except Exception:  # noqa: BLE001
            self._inflight.release()
            return None
        if not done.wait(self.timeout_s) or box.get("at", float("inf")) - started > self.timeout_s:
            return None
        return box.get("p")


def looking_back_for(settings) -> LookingBackCheck | None:
    """On the reflex's TypeSafe key, like the honesty judge; None without one (then no turn is ever moved)."""
    key = getattr(settings, "typesafe_api_key", None)
    if not key:
        return None
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    client = TypeSafeClient(api_key=key.get_secret_value(), model=MODEL, timeout=TIMEOUT_S,
                            retry=RetryPolicy(max_retries=0))
    return LookingBackCheck(client)
