"""Speaker redirect — a satellite whose speaker is gone speaks through another machine.

Owner ask 2026-09-26: the office satellite's speaker died (a kitten), and "if she is at
my desk, does she really need her own speaker? could she inject audio through
leviathan's default output?". For a device listed in VOICE_SPEAKER_REDIRECT, every
reply that would have been spoken on that satellite — the /ask reply and the spoken
follow-ups — is ALSO rendered by Home Assistant in her pipeline voice
(``/api/tts_get_url``, same engine + voice as the Assist pipeline) and the audio URL is
handed to a small player service on the target machine (aerys-speaker on Leviathan).

The satellite still gets the reply as before (dead speaker = silent, so no double
voice); remove the device from the map when a working speaker is back. Fire-and-forget
on a daemon thread; every failure is a debug log — a dark desk speaker costs nothing.
"""

from __future__ import annotations

import contextvars
import logging
import re
import threading
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

#: emotion tags for ElevenLabs v3 stay in (her pipeline voice reads them); markdown
#: does not — asterisks and backticks have no business in speech
_MARKDOWN = re.compile(r"(\*\*|__|`+|^#+\s*)", re.M)

#: Chris 11/13 (the office button, "she shuts up"): a hush ends everything she was going to say there
#: for the turns that began before it: the clip playing, what's queued, a clip still being made, a
#: research answer still coming, a mic reopen. A turn notes when it began (turn_started, carried to its
#: background follow-up through the copied context); every clip carries that time ("since"), and the
#: speaker itself refuses a clip from before its last hush ("hush_at" on /stop). The speaker is the
#: judge, so no race between connections can let a hushed clip through, and a newer clip is never
#: stopped by an older one. Speech that belongs to no turn (the reach desk) is "since now": it plays.
#: Wall-clock seconds from this brain on both sides, so a restart doesn't reset anything.
_HUSH_AT: dict[str, float] = {}
_HUSH_LOCK = threading.Lock()
_TURN: contextvars.ContextVar[float | None] = contextvars.ContextVar("speaker_turn", default=None)


def hush(device_id: str | None) -> float:
    """Hush this speaker as of now; returns the time, for the speaker's own floor."""
    now = time.time()
    if device_id:
        with _HUSH_LOCK:
            _HUSH_AT[str(device_id)] = max(_HUSH_AT.get(str(device_id), 0.0), now)
    return now


def hushed_at(device_id: str | None) -> float:
    with _HUSH_LOCK:
        return _HUSH_AT.get(str(device_id or ""), 0.0)


def turn_started(device_id: str | None = None) -> None:
    """A voice turn began now: whatever it says later, on any speaker, is from this turn."""
    _TURN.set(time.time())


def _since() -> float:
    turn = _TURN.get()
    return turn if turn is not None else time.time()


def parse_targets(raw: str) -> dict[str, str]:
    """'device_id=http://host:port,…' -> {device_id: base_url}."""
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        dev, sep, url = part.strip().partition("=")
        if sep and dev.strip() and url.strip():
            out[dev.strip()] = url.strip().rstrip("/")
    return out


class SpeakerRedirect:
    def __init__(self, *, ha_base_url: str, ha_token: str, engine: str, voice: str, language: str,
                 targets: dict[str, str], speaker_token: str, client=None, async_send: bool = True) -> None:
        import httpx

        self._ha = ha_base_url.rstrip("/")
        self._ha_headers = {"Authorization": f"Bearer {ha_token}"}
        self._engine, self._voice, self._language = engine, voice, language
        self._targets = targets
        self._speaker_headers = {"Authorization": f"Bearer {speaker_token}"}
        self._client = client or httpx.Client(timeout=30.0)
        self._async = async_send

    def handles(self, device_id: str | None) -> bool:
        return bool(device_id) and device_id in self._targets

    def playing(self, device_id: str | None) -> bool:
        """Best-effort: is she talking there, a clip playing or more queued behind it? Failures read
        as not talking (harmless). 11/13: "On it." queued behind a long answer is still her talking."""
        if not self.handles(device_id):
            return False
        try:
            response = self._client.get(
                f"{self._targets[device_id]}/health",
                headers=self._speaker_headers, timeout=1.5,
            )
            response.raise_for_status()
            health = response.json()
            return health.get("playing") is True or int(health.get("queued") or 0) > 0
        except Exception:
            log.debug("speaker redirect playing failed (harmless)", exc_info=True)
            return False

    def stop(self, device_id: str | None, hush_at: float | None = None) -> bool:
        """Best-effort speaker stop; failures are harmless. With hush_at the speaker also refuses
        every later clip from a turn that began before it."""
        if not self.handles(device_id):
            return False
        try:
            response = self._client.post(
                f"{self._targets[device_id]}/stop",
                headers=self._speaker_headers, timeout=3.0,
                json={"hush_at": hush_at} if hush_at is not None else {},
            )
            response.raise_for_status()
            return response.json().get("stopped") is True
        except Exception:
            log.debug("speaker redirect stop failed (harmless)", exc_info=True)
            return False

    def say(self, device_id: str | None, text: str, *, then: Callable[[], None] | None = None) -> None:
        if not self.handles(device_id) or not (text or "").strip():
            return
        since = _since()
        if hushed_at(device_id) > since:
            log.info("speaker redirect: hushed on %s since this turn began; not saying it", device_id)
            return
        if self._async:
            threading.Thread(target=self._say, args=(device_id, text, then, since), daemon=True).start()
        else:
            self._say(device_id, text, then, since)

    def _say(self, device_id: str, text: str, then: Callable[[], None] | None = None,
             since: float | None = None) -> None:
        since = time.time() if since is None else since
        try:
            r = self._client.post(
                f"{self._ha}/api/tts_get_url",
                headers=self._ha_headers,
                json={"engine_id": self._engine, "message": _MARKDOWN.sub("", text),
                      "language": self._language, "options": {"voice": self._voice}},
            )
            r.raise_for_status()
            url = r.json()["url"]
            if hushed_at(device_id) > since:
                return                       # hushed while it was being made: it never plays
            p = self._client.post(f"{self._targets[device_id]}/play", headers=self._speaker_headers,
                                  json={"url": url, "since": since, **({"wait": True} if then is not None else {})},
                                  timeout=130.0 if then is not None else 5.0)
            p.raise_for_status()
            if then is not None and p.json().get("played") is True:
                then()
        except Exception:
            log.debug("speaker redirect failed (harmless)", exc_info=True)


def speaker_redirect_for(settings) -> SpeakerRedirect | None:
    """Armed only when HA, a target map, the speaker token and the pipeline voice are
    all configured — the standard optional-seam pattern."""
    targets = parse_targets(getattr(settings, "voice_speaker_redirect", ""))
    token = getattr(settings, "voice_speaker_token", None)
    if not targets or token is None or settings.ha_token is None or not settings.ha_tts_voice:
        return None
    return SpeakerRedirect(
        ha_base_url=settings.ha_base_url, ha_token=settings.ha_token.get_secret_value(),
        engine=settings.ha_tts_engine, voice=settings.ha_tts_voice, language=settings.ha_tts_language,
        targets=targets, speaker_token=token.get_secret_value(),
    )
