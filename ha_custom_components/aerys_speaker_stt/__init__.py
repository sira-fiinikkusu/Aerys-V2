"""Aerys Speaker STT — who said it, without touching what was said.

A speech-to-text entity that WRAPS an existing STT entity (Home Assistant Cloud by
default): the audio stream passes through to the source recognizer unchanged, a copy
is buffered, and when the stream ends the clip goes to the aerys-voice-id recognizer
on the Jetson (POST /recognize) CONCURRENTLY with the cloud transcript. The speaker
verdict lands in hass.data for the aerys_conversation agent to carry on the /ask body.

Forked from EuleMitKeule/speaker-recognition's stt.py (MIT) — owner ruling 2026-09-19:
cloud STT stays; identity comes from the voice, not the puck. Fail-open on identity,
never on speech: any recognizer trouble means an untagged turn, the transcript is
never delayed past `recognizer_timeout`.
"""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN

PLATFORMS = [Platform.STT]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    hass.data.setdefault(DOMAIN, {})
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_reload))
    return True


async def _async_reload(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
