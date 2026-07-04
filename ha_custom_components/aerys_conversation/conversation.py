"""The Aerys conversation agent — a pure forwarder to the aerys-v2 Brain.

It carries ``user_input.device_id`` (the originating satellite, populated by the
Assist pipeline) into the ``/ask`` body so the Brain can route the spoken
follow-up back to the SAME device. No LLM work happens here.
"""

from __future__ import annotations

from typing import Literal

import aiohttp

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import CONF_API_TOKEN, CONF_BASE_URL

# One shared voice thread for the beta pipeline (owner decision: voice rides the
# owner thread) — matches the thread_id the OpenAI shim used, so history is
# continuous across the pipeline swap.
THREAD_ID = "voice:beta"
DISPLAY_NAME = "Chris (Voice)"
REQUEST_TIMEOUT_S = 15


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Aerys conversation agent from a config entry."""
    async_add_entities([AerysConversationAgent(config_entry)])


class AerysConversationAgent(ConversationEntity, conversation.AbstractConversationAgent):
    """Forwards HA Assist turns to aerys-v2's /ask, carrying the device_id."""

    _attr_has_entity_name = True
    _attr_name = "Aerys"
    # CONTROL: the Brain runs its OWN gated home-control path (HA_CANARY_ENTITIES),
    # so this agent is allowed to affect device state on the owner's behalf.
    _attr_supported_features = ConversationEntityFeature.CONTROL

    def __init__(self, entry: ConfigEntry) -> None:
        self.entry = entry
        # base_url + api_token come from the config entry — never hardcoded here.
        self._base_url = entry.data[CONF_BASE_URL].rstrip("/")
        self._api_token = entry.data[CONF_API_TOKEN]
        self._attr_unique_id = entry.entry_id

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    async def _async_handle_message(
        self, user_input: ConversationInput, chat_log
    ) -> ConversationResult:
        session = async_get_clientsession(self.hass)
        intent_response = intent.IntentResponse(language=user_input.language)
        try:
            resp = await session.post(
                f"{self._base_url}/ask",
                json={
                    "text": user_input.text,
                    "thread_id": THREAD_ID,
                    "display_name": DISPLAY_NAME,
                    # The load-bearing field: which satellite started this turn.
                    "device_id": user_input.device_id,
                },
                headers={"Authorization": f"Bearer {self._api_token}"},
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
            )
            resp.raise_for_status()
            data = await resp.json()
            intent_response.async_set_speech(data["reply"])
        except Exception as err:  # never crash the pipeline — speak the failure
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Aerys is unreachable: {err}",
            )
        return ConversationResult(
            response=intent_response, conversation_id=user_input.conversation_id
        )
