"""#261 (11/25): a judged "does this fact say what they said?" check, beside the strict word check in remember.

D105 made a remembered fact "their words" (unmarked in recall) only when the word check proves it. That check is
lexical: every content word of the fact must be in what they said. It can't see a paraphrase ("I'm up at 6" ->
"Chris gets up at 6" stays hers), and it can't see a change of roles or certainty with the same words ("Bob trusts
Alice" -> "Alice trusts Bob", "I might like coffee" -> "I like coffee" pass it). Jev answers three judgements about
the fact and their message; code decides (the TypeSafe rule: judgements from Jev, decisions in code).

Wired in remember._judged: a clear veto (it adds or changes something) turns the word check's "their words" back
into hers; and (D107, the owner 11/25 22:21: "we can give Jev a lot of agency here") a clear "theirs" (a restatement
that adds and changes nothing) makes a paraphrase their words, past the strict word check.
Any failure (timeout, error, no key, a bad score, no SDK) is None: the word check stands alone, exactly as before.
"""
from __future__ import annotations

import contextvars
import math
import threading
import time
from dataclasses import dataclass

MODEL = "jev-1.14.0"
TIMEOUT_S = 1.5
MESSAGE_MAX = 1200
FACT_MAX = 500
YES_AT = 0.8          # restates
NO_AT = 0.2           # adds / changes, for "their words"
VETO_AT = 0.5         # adds / changes, for a veto

QUESTIONS = {
    "restates": {
        "type": "noul",
        "instructions": "Does `fact` say something that `speaker` said in `their_message`?",
        "criteria": {
            "true": "The fact restates what they said, in the same or other words, possibly naming `speaker` in "
                    "place of I or my (\"I'm up at 6 most days\" -> \"Sam gets up at 6 most days\" when the speaker "
                    "is Sam).",
            "false": "The fact is about something they didn't say in this message (\"I got a new phone\" -> \"Chris "
                     "likes Samsung phones\").",
        },
    },
    "adds": {
        "type": "noul",
        "instructions": "Does `fact` include a detail that `their_message` doesn't contain?",
        "criteria": {
            "true": "The fact has a name, number, date, place, reason, or conclusion that isn't in their message "
                    "(\"I take vitamins\" -> \"Sam takes vitamin D\"; \"I'm going to Denver\" -> \"Sam is going "
                    "to Denver for work\").",
            "false": "Every detail in the fact is in their message. Naming `speaker` in place of I, me or my is "
                     "not an added detail, and neither is a date written another way (\"Dec 19\" / \"December 19\").",
        },
    },
    "changes": {
        "type": "noul",
        "instructions": "Does `fact` change the meaning of what `their_message` says?",
        "criteria": {
            "true": "The fact flips a yes into a no or a no into a yes, swaps who did what, changes a number, or "
                    "makes a maybe certain (\"I don't like coffee\" -> \"likes coffee\"; \"Bob trusts Alice\" -> "
                    "\"Alice trusts Bob\"; \"I might move\" -> \"is moving\").",
            "false": "The fact keeps their meaning: the same polarity, the same roles, the same numbers and the same "
                     "certainty.",
        },
    },
}


@dataclass(frozen=True)
class Judgement:
    restates: float
    adds: float
    changes: float
    # Codex 11/26: False when the message or fact was clipped for Jev. A cut-off tail may retract what the head
    # says ("I like coffee ... Correction: I do not"), so a clipped judgement never upgrades; a veto still counts.
    complete: bool = True

    @property
    def theirs(self) -> bool:
        return (self.complete and self.restates >= YES_AT and self.adds <= NO_AT
                and self.changes <= NO_AT)

    @property
    def veto(self) -> bool:
        return self.adds >= VETO_AT or self.changes >= VETO_AT


def _flat(text) -> str:
    return " ".join(str(text or "").split())


def fits(fact: str, their_message: str) -> bool:
    """Both texts reach Jev whole (nothing clipped)."""
    return len(_flat(their_message)) <= MESSAGE_MAX and len(_flat(fact)) <= FACT_MAX


def build_state(fact: str, their_message: str, speaker: str = "") -> dict:
    def clip(text: str, limit: int) -> str:
        text = " ".join(str(text or "").split())
        return text if len(text) <= limit else text[:limit] + "…"
    return {"speaker": clip(speaker, 60) or "the speaker", "their_message": clip(their_message, MESSAGE_MAX),
            "fact": clip(fact, FACT_MAX)}


def _score(v) -> float | None:
    return float(v) if (not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                        and 0.0 <= v <= 1.0) else None


class TheirWordsCheck:
    """check(fact, their_message) -> Judgement, or None (unavailable). One bounded request; a late answer is
    never used."""

    def __init__(self, client: object, *, model: str = MODEL, timeout_s: float = TIMEOUT_S):
        self.client, self.model, self.timeout_s = client, model, timeout_s
        self._inflight = threading.BoundedSemaphore(4)

    def __call__(self, fact: str, their_message: str, speaker: str = "") -> Judgement | None:
        state = build_state(fact, their_message, speaker)
        if not state["fact"] or not state["their_message"]:
            return None
        started = time.monotonic()
        box: dict = {}
        done = threading.Event()

        def run():
            try:
                response = self.client.system_one(state=state, questions=QUESTIONS, model=self.model)
                scores = [_score(response.answers[q].noul) for q in ("restates", "adds", "changes")]
                if all(s is not None for s in scores):
                    box["j"] = Judgement(*scores, complete=fits(fact, their_message))
            except Exception:  # noqa: BLE001 -- the check never blocks a write
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
        return box.get("j")


def their_words_for(settings) -> TheirWordsCheck | None:
    """On the reflex's TypeSafe key, like looking_back; None without one (then the word check stands alone)."""
    key = getattr(settings, "typesafe_api_key", None)
    if not key:
        return None
    try:
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        client = TypeSafeClient(api_key=key.get_secret_value(), model=MODEL, timeout=TIMEOUT_S,
                                retry=RetryPolicy(max_retries=0))
    except Exception:  # noqa: BLE001 -- Gemini review: a missing/broken SDK means no veto, never a failed build
        import logging
        logging.getLogger(__name__).warning("their-words check unavailable; the word check stands alone", exc_info=True)
        return None
    return TheirWordsCheck(client)
