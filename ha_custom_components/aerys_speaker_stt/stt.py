"""The wrapping STT entity (see package docstring)."""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections.abc import AsyncIterable

import aiohttp

from homeassistant.components.stt import (
    AudioBitRates, AudioChannels, AudioCodecs, AudioFormats, AudioSampleRates,
    SpeechMetadata, SpeechResult, SpeechResultState, SpeechToTextEntity,
    async_get_speech_to_text_entity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event

from .const import (
    CONF_RECOGNIZER_URL, CONF_SOURCE_STT, CONF_TIMEOUT, DEFAULT_TIMEOUT, DOMAIN, LAST_KEY,
)

_LOGGER = logging.getLogger(__name__)
EVENT_SPEAKER = "aerys_speaker_detected"


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    async_add_entities([AerysSpeakerSttEntity(hass, entry)])


class AerysSpeakerSttEntity(SpeechToTextEntity):
    _attr_should_poll = False
    _attr_has_entity_name = False

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._source_id: str = entry.data[CONF_SOURCE_STT]
        self._url: str = entry.data[CONF_RECOGNIZER_URL].rstrip("/")
        self._timeout: float = float(entry.data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))
        self._attr_unique_id = entry.entry_id
        self._attr_name = f"Aerys speaker ({self._source_id.split('.', 1)[-1]})"

    # ---- mirror the source entity's capabilities -------------------------------
    def _source(self):
        return async_get_speech_to_text_entity(self.hass, self._source_id)

    @property
    def supported_languages(self) -> list[str]:
        s = self._source(); return list(s.supported_languages) if s else []

    @property
    def supported_formats(self) -> list[AudioFormats]:
        s = self._source(); return list(s.supported_formats) if s else []

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        s = self._source(); return list(s.supported_codecs) if s else []

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        s = self._source(); return list(s.supported_bit_rates) if s else []

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        s = self._source(); return list(s.supported_sample_rates) if s else []

    @property
    def supported_channels(self) -> list[AudioChannels]:
        s = self._source(); return list(s.supported_channels) if s else []

    @callback
    def _source_changed(self, event: Event[EventStateChangedData] | None = None) -> None:
        state = self.hass.states.get(self._source_id)
        self._attr_available = state is not None and state.state != STATE_UNAVAILABLE
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(async_track_state_change_event(self.hass, [self._source_id], self._source_changed))
        self._source_changed()

    # ---- the turn ---------------------------------------------------------------
    async def async_process_audio_stream(self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]) -> SpeechResult:
        source = self._source()
        if source is None:
            _LOGGER.error("source STT entity %s is missing", self._source_id)
            return SpeechResult(None, SpeechResultState.ERROR)

        buffer = bytearray()
        recognize: asyncio.Task | None = None
        loop = asyncio.get_running_loop()

        async def tee() -> AsyncIterable[bytes]:
            nonlocal recognize
            async for chunk in stream:
                buffer.extend(chunk)
                yield chunk
            # Stream over: the clip is complete, start recognition NOW so it runs while
            # the cloud recognizer is still finishing its transcript.
            if buffer:
                recognize = loop.create_task(self._recognize(bytes(buffer), metadata))

        result = await source.async_process_audio_stream(metadata, tee())

        if recognize is not None:
            try:
                verdict = await asyncio.wait_for(recognize, timeout=self._timeout)
            except asyncio.TimeoutError:
                _LOGGER.warning("speaker recognizer timed out (%.1fs); turn untagged", self._timeout)
                verdict = None
            except Exception:  # noqa: BLE001 — identity is fail-open, speech is not
                _LOGGER.warning("speaker recognizer failed; turn untagged", exc_info=True)
                verdict = None
            if verdict:
                self.hass.data.setdefault(DOMAIN, {})[LAST_KEY] = {**verdict, "at": time.monotonic(), "entity_id": self.entity_id}
                self.hass.bus.async_fire(EVENT_SPEAKER, {**verdict, "entity_id": self.entity_id})
                _LOGGER.info("speaker: %s (%.2f) %s", verdict["user_id"], verdict["confidence"], verdict.get("all_scores"))
        return result

    async def _recognize(self, audio: bytes, metadata: SpeechMetadata) -> dict | None:
        # PCM16 mono is what the satellites send; anything else is passed through untagged.
        if metadata.codec != AudioCodecs.PCM or metadata.channel != AudioChannels.CHANNEL_MONO:
            return None
        session = async_get_clientsession(self.hass)
        body = {"audio": {"audio_data": base64.b64encode(audio).decode(), "sample_rate": int(metadata.sample_rate)}}
        async with session.post(f"{self._url}/recognize", json=body,
                                timeout=aiohttp.ClientTimeout(total=self._timeout + 0.5)) as resp:
            if resp.status == 409:  # nobody enrolled yet — untagged, by design
                return None
            resp.raise_for_status()
            data = await resp.json()
        return {"user_id": str(data.get("user_id") or "unknown"), "confidence": float(data.get("confidence") or 0.0),
                "all_scores": data.get("all_scores") or {}}
