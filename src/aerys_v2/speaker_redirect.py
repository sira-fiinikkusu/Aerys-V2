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

import logging
import re
import threading

log = logging.getLogger(__name__)

#: emotion tags for ElevenLabs v3 stay in (her pipeline voice reads them); markdown
#: does not — asterisks and backticks have no business in speech
_MARKDOWN = re.compile(r"(\*\*|__|`+|^#+\s*)", re.M)


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

    def say(self, device_id: str | None, text: str) -> None:
        if not self.handles(device_id) or not (text or "").strip():
            return
        if self._async:
            threading.Thread(target=self._say, args=(device_id, text), daemon=True).start()
        else:
            self._say(device_id, text)

    def _say(self, device_id: str, text: str) -> None:
        try:
            r = self._client.post(
                f"{self._ha}/api/tts_get_url",
                headers=self._ha_headers,
                json={"engine_id": self._engine, "message": _MARKDOWN.sub("", text),
                      "language": self._language, "options": {"voice": self._voice}},
            )
            r.raise_for_status()
            url = r.json()["url"]
            p = self._client.post(f"{self._targets[device_id]}/play", headers=self._speaker_headers,
                                  json={"url": url}, timeout=5.0)
            p.raise_for_status()
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
