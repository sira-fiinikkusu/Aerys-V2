"""D90 (the owner, 11/22 13:34): a LENIENT "was that for me?" on his glasses' follow-up turns.

11/22 18:26: Even AI's Continuous listening sent nearby speech ("We didn't tour.", "You want?") to her as his
turns every 8-12 s, and the lens path's own unaddressed-drop stands aside while a conversation is in flight
(so it can't cut him off mid-chat). His words: "Lets.. try the was that for me change, I just dont want it to
be over strict is all". So this check:
  - runs only on his lens follow-ups (a conversation in flight); a first turn after "Hey Even" always answers;
  - asks Jev ONE judgement: were these words said to her, or by people nearby;
  - keeps her quiet only when Jev is very sure they weren't (SKIP_BELOW); everything else is answered;
  - never drops on doubt: a timeout, an error or a bad score answers the turn (lenient by construction).
Judgements from Jev, the decision in code (the TypeSafe rule). The state is code-written from the thread's own
last exchange; nothing else is sent. The real fix is the Enterprise SDK's speaker role (his voice only).
"""
from __future__ import annotations

import contextvars
import math
import threading
import time

MODEL = "jev-1.14.0"
TIMEOUT_S = 1.0
#: Lenient: she stays quiet only when Jev gives the words less than this chance of being said to her.
SKIP_BELOW = 0.15
#: Bounded state: a long reply keeps its end (where her question to him usually is).
SHE_SAID_MAX, HEARD_MAX, BEFORE_MAX = 600, 400, 400

QUESTION_ID = "said_to_her"
QUESTIONS = {
    QUESTION_ID: {
        "type": "noul",
        "instructions": (
            "The glasses' microphone picked up `heard` during a conversation with the assistant (`she_said` is "
            "the assistant's last reply; `he_said_before` is what the wearer said to it just before). "
            "Were the words in `heard` said to the assistant?"
        ),
        "criteria": {
            "true": "The wearer is talking to the assistant: a question, a request, or an answer or follow-up "
                    "to what the assistant just said (for example 'yes, that one', 'what time does it open?', "
                    "or a short reply to a question it asked).",
            "false": "Someone nearby is talking to another person, and it doesn't follow from what the assistant "
                     "said (for example 'babe, grab the tickets from the car' or a remark about something else).",
        },
    }
}


def _tail(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else "…" + text[-n:]


def build_state(heard: str, she_said: str, he_said_before: str) -> dict:
    return {"heard": _tail(heard, HEARD_MAX), "she_said": _tail(she_said, SHE_SAID_MAX),
            "he_said_before": _tail(he_said_before, BEFORE_MAX)}


def skip(p: float | None) -> bool:
    """Stay quiet? Only on a real score below SKIP_BELOW. None (no answer in time) always answers."""
    return p is not None and p < SKIP_BELOW


class AddressedCheck:
    """check(heard, she_said, he_said_before) -> the chance the words were said to her, or None (unavailable:
    timeout, error, bad score). One request, bounded; a late answer is never used."""

    def __init__(self, client: object, *, model: str = MODEL, timeout_s: float = TIMEOUT_S):
        self.client, self.model, self.timeout_s = client, model, timeout_s
        self._inflight = threading.BoundedSemaphore(8)

    def __call__(self, heard: str, she_said: str, he_said_before: str) -> float | None:
        started = time.monotonic()
        state = build_state(heard, she_said, he_said_before)
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


def check_for(settings) -> AddressedCheck | None:
    """On the reflex's TypeSafe key, like the honesty judge; None without one (then nothing is ever skipped)."""
    key = getattr(settings, "typesafe_api_key", None)
    if not key:
        return None
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    client = TypeSafeClient(api_key=key.get_secret_value(), model=MODEL, timeout=TIMEOUT_S,
                            retry=RetryPolicy(max_retries=0))
    return AddressedCheck(client)
