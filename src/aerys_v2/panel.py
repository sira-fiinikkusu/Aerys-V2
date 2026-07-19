"""The panel face seam — her desk avatar mirrors what the brain is doing.

The reTerminal panel is deliberately dumb: it loops one MJPEG face state and
exposes ``POST /state`` to swap it. This module is the brain-side half: a
single fire-and-forget callable the service layer pokes at three phases of a
turn — ``working`` (tools are grinding), ``speaking`` (words are leaving for a
speaker), ``idle`` (the turn settled). The mood baked into the reply's
ElevenLabs emotion tags picks WHICH face; the phase picks idle-vs-speaking.

Design constraints, in order:
- NEVER touch the hot path: every HTTP send happens on a daemon thread with a
  short timeout, and every failure is swallowed to a debug log. A dark panel
  must cost nothing; a slow panel must cost nothing.
- The panel owns no timing. TTS playback duration is invisible to the brain
  (the voice pipeline speaks the returned text after we've moved on), so a
  ``speaking`` push schedules its own flip back to the idle face after a
  length-based estimate — cancelled the moment any newer push lands.
- Unknown states 404 on the panel and she keeps her current face; a missing
  clip is cosmetic, never an error.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

# Face states installed on the panel's SD card (finish.sh -> /faces/*.mjpeg).
# Moods without a speaking variant fall back to the nearest one that reads
# right in motion: playful speaks as happy, grumpy speaks as neutral (deadpan).
_IDLE = {
    "neutral": "neutral_idle",
    "happy": "happy_idle",
    "playful": "playful_idle",
    "grumpy": "grumpy_idle",
    "surprised": "surprised",
    "affection": "heart_emote",
}
_SPEAKING = {
    "neutral": "neutral_speaking",
    "happy": "happy_speaking",
    "playful": "happy_speaking",
    "grumpy": "neutral_speaking",
    "surprised": "surprised",
    "affection": "heart_emote",
}
WORKING_STATE = "working"

# ElevenLabs v3 emotion tags -> mood. First recognized tag in the text wins —
# the polisher leads with the dominant emotion, so first is most representative.
_TAG_MOODS = {
    # neutral is mapped explicitly so a leading calm tag WINS over a later
    # colorful one — the polisher leads with the dominant emotion.
    "neutral": ("softly", "calmly", "gently", "thoughtfully", "quietly", "evenly"),
    "happy": (
        "warmly", "happily", "excited", "excitedly", "cheerfully", "laughs",
        "laughing", "giggles", "giggling", "delighted", "brightly", "chuckles",
    ),
    "playful": ("playfully", "teasing", "teasingly", "mischievously", "smirks", "slyly"),
    "grumpy": (
        "annoyed", "frustrated", "sarcastically", "grumbles", "sighs",
        "deadpan", "flatly", "exasperated",
    ),
    "surprised": ("surprised", "gasps", "shocked", "amazed", "astonished"),
    "affection": ("lovingly", "affectionately", "tenderly", "adoringly"),
}
_WORD_TO_MOOD = {
    word: mood for mood, words in _TAG_MOODS.items() for word in words
}
_TAG_RE = re.compile(r"\[([a-z][a-z ]*)\]")
_HEART_RE = re.compile("[❤\U0001f49a\U0001f496\U0001f5a4\U0001f49c\U0001f970\U0001f60d\U0001f497]")


def mood_of(text: str) -> str:
    """Extract the mood a reply should wear on the panel.

    Emotion tags only appear on voice-styled replies; text replies usually
    carry none and read neutral — except hearts, which are unambiguous enough
    to earn the heart_emote on any channel.
    """
    for match in _TAG_RE.finditer(text or ""):
        for word in match.group(1).split():
            mood = _WORD_TO_MOOD.get(word)
            if mood is not None:
                return mood
    if _HEART_RE.search(text or ""):
        return "affection"
    return "neutral"


def speaking_estimate_s(text: str) -> float:
    """Rough TTS playback duration for the speaking->idle auto-flip.

    Tags are stripped first (spoken silently as emotion). ~16-18 chars/sec is
    typical conversational TTS; clamp so one-word acks still hold the speaking
    face a beat and rambles don't pin it for a minute.
    """
    visible = _TAG_RE.sub("", text or "")
    return max(2.5, min(20.0, 1.5 + len(visible) / 16.0))


class FacePusher:
    """Callable seam: ``push(phase, text)`` with phase in working|speaking|idle.

    Consecutive duplicate states are skipped (a chat flurry shouldn't restart
    her idle loop), sends ride daemon threads, and a speaking push arms a timer
    that settles her back to the matching idle face — any newer push cancels it.
    """

    def __init__(self, url: str, *, client=None, async_send: bool = True) -> None:
        import httpx

        self._url = url
        self._client = client or httpx.Client(timeout=2.0)
        self._async = async_send
        self._lock = threading.Lock()
        self._last_state: str | None = None
        self._gen = 0
        self._timer: threading.Timer | None = None
        self._timer_fires_at = 0.0

    def __call__(self, phase: str, text: str = "") -> None:
        try:
            mood = mood_of(text)
            if phase == "working":
                state = WORKING_STATE
            elif phase == "speaking":
                state = _SPEAKING.get(mood, "neutral_speaking")
            else:
                state = _IDLE.get(mood, "neutral_idle")

            now = time.monotonic()
            with self._lock:
                if phase == "working" and self._timer_fires_at > now:
                    # She's mid-ack ("On it — one sec"): let the speaking face
                    # finish its estimated run, THEN switch to the working face.
                    # The pending flip is re-aimed at working instead of idle.
                    self._arm_timer(self._timer_fires_at - now, WORKING_STATE)
                    return
                self._gen += 1      # invalidate any pending deferred send
                self._cancel_timer()
                dup = state == self._last_state
                self._last_state = state
                if phase == "speaking":
                    # Settle back to this mood's idle face when the words run out.
                    self._arm_timer(
                        speaking_estimate_s(text), _IDLE.get(mood, "neutral_idle")
                    )
            if not dup:
                self._send(state)
        except Exception:  # the panel must never cost a turn anything
            log.debug("face push failed (harmless)", exc_info=True)

    def _arm_timer(self, delay: float, state: str) -> None:
        # Caller holds the lock. One pending deferred send at a time.
        if self._timer is not None:
            self._timer.cancel()
        self._gen += 1
        timer = threading.Timer(delay, self._deferred_send, (self._gen, state))
        timer.daemon = True
        timer.start()
        self._timer = timer
        self._timer_fires_at = time.monotonic() + delay

    def _cancel_timer(self) -> None:
        # Caller holds the lock.
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._timer_fires_at = 0.0

    def _deferred_send(self, gen: int, state: str) -> None:
        try:
            with self._lock:
                if gen != self._gen:
                    return  # a newer push owns the face now
                self._timer = None
                self._timer_fires_at = 0.0
                if state == self._last_state:
                    return
                self._last_state = state
            self._send(state)
        except Exception:
            log.debug("face deferred push failed (harmless)", exc_info=True)

    def _send(self, state: str) -> None:
        if self._async:
            threading.Thread(
                target=self._post, args=(state,), daemon=True
            ).start()
        else:
            self._post(state)

    def _post(self, state: str) -> None:
        try:
            self._client.post(self._url, json={"state": state})
        except Exception:
            log.debug("panel unreachable (harmless)", exc_info=True)


def build_face_pusher(
    url: str | None, *, client=None, async_send: bool = True
) -> Callable[[str, str], None] | None:
    """None unless a panel URL is configured — the standard optional-seam arming
    pattern. The URL lives in the environment (PANEL_STATE_URL), never in code:
    it's a LAN address and this repo is public."""
    if not url:
        return None
    return FacePusher(url, client=client, async_send=async_send)
