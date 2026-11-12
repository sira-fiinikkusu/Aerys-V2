"""Fail-closed HA toggle and process-local, device-scoped voice follow-up windows."""

import threading
import time
import unicodedata

import httpx


class ToggleReader:
    """Cache even failed reads for five seconds; never log credentials or HA bodies."""

    def __init__(self, settings):
        self._settings = settings
        self._lock = threading.Lock()
        self._expires = 0.0
        self._enabled = False

    def enabled(self) -> bool:
        # The HA read happens OUTSIDE the lock (Gemini review 11/11): a slow HA never makes one voice
        # turn wait on another's read; at worst two turns read it at the same time.
        with self._lock:
            if time.monotonic() < self._expires:
                return self._enabled
        enabled = False
        try:
            settings = self._settings
            if settings is not None and settings.ha_token is not None:
                response = httpx.get(
                    settings.ha_base_url.rstrip("/") + "/api/states/input_boolean.aerys_keep_listening",
                    headers={"Authorization": f"Bearer {settings.ha_token.get_secret_value()}"},
                    timeout=2.0,
                )
                response.raise_for_status()
                enabled = response.json().get("state") == "on"
        except Exception:
            # Missing toggle, timeout, malformed response, or HA unavailable = OFF.
            pass
        with self._lock:
            self._enabled = enabled
            self._expires = time.monotonic() + 5.0
        return enabled


opened_mic: dict[str, float] = {}
_mic_lock = threading.Lock()
_STOP_WORDS = {"stop", "never mind", "nevermind", "cancel", "thats all"}


def mark_opened(device_id: str | None) -> None:
    if not device_id:
        return
    with _mic_lock:
        now = time.monotonic()
        for device in list(opened_mic):
            if now - opened_mic[device] > 30.0:
                del opened_mic[device]
        opened_mic[device_id] = now


def consume_dismissal(device_id: str | None, text: str) -> bool:
    """The next voice input consumes this window, whether an answer or dismissal.

    Missing device IDs never share a window. Pop under the lock so concurrent
    inputs cannot both silently dismiss the same open mic.
    """
    if not device_id:
        return False
    with _mic_lock:
        opened = opened_mic.pop(device_id, None)
    normalized = " ".join("".join(
        ch for ch in text.lower() if not unicodedata.category(ch).startswith("P")
    ).split())
    return (opened is not None and 0 <= time.monotonic() - opened <= 30.0
            and normalized in _STOP_WORDS)


ENTITY = "input_boolean.aerys_keep_listening"


def build_keep_listening_tool(settings, *, post=None):
    """Her voice switch for the owner's toggle (11/11: "can it be a voice toggle too? ... say im in the kitchen
    and up for a chat"). Flips the HA helper; HA turns it off by itself after 2 hours or at his bedtime."""
    from langchain_core.tools import tool

    send = post or httpx.post

    @tool
    def keep_listening(on: bool) -> str:
        """Turn "keep listening" on or off: when it's on, after you ask him something out loud the mic stays
        open in that room so he can just answer.

        CALL THIS with on=true when he says he's up for a chat, "keep listening", "you can keep listening",
        "let's talk for a bit". CALL THIS with on=false when he says "stop listening", "that's enough",
        "you can stop listening", "I'm done chatting". It turns itself off after 2 hours or when he goes to bed.
        Tell him plainly which it is now.
        """
        if settings is None or settings.ha_token is None:
            return "Couldn't change it: I can't reach Home Assistant from here."
        service = "turn_on" if on else "turn_off"
        try:
            response = send(f"{settings.ha_base_url.rstrip('/')}/api/services/input_boolean/{service}",
                            headers={"Authorization": f"Bearer {settings.ha_token.get_secret_value()}"},
                            json={"entity_id": ENTITY}, timeout=10.0)
            response.raise_for_status()
        except Exception:
            return "Couldn't change it: Home Assistant didn't take it. Nothing changed."
        return ("Keep listening is on: after I ask you something, I'll keep listening for your answer. It turns "
                "itself off after 2 hours or at bedtime." if on else
                "Keep listening is off: I'll speak and stop, like before.")

    return keep_listening
