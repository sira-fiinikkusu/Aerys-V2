"""The Aerys Conversation integration.

A thin transport that runs INSIDE Home Assistant: it forwards each Assist turn to
the aerys-v2 Brain's ``/ask`` endpoint and speaks back whatever ``reply`` comes
back. It does zero LLM work — aerys-v2 owns all the intelligence.

Why it lives in HA instead of the Brain: only HA sees
``ConversationInput.device_id`` (the originating satellite), populated by the
Assist pipeline before any conversation agent runs. This component rides that
device_id on the outbound ``/ask`` request so the Brain's spoken follow-up
answers on the SAME device the voice turn came from — the whole reason this
component replaces the OpenAI-shim path (which had no field for device identity).

Same "normalize -> ask() -> reply" shape as every other aerys-v2 transport
(transports/http_api.py, transports/discord_gateway.py).

INSTALL (not done by this repo — see docs/satellite-routing-design.md §3b):
drop this folder at ``/config/custom_components/aerys_conversation/`` on HA
Green, restart HA, then point the "Aerys-beta" pipeline's conversation agent at
this component. Do NOT touch the "Aerys" (stable) pipeline — that is still the
production n8n path.
"""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

DOMAIN = "aerys_conversation"

# Config-entry keys — base_url and api_token are supplied by the config flow, so
# NOTHING (least of all the Bearer token) is hardcoded in source.
CONF_BASE_URL = "base_url"
CONF_API_TOKEN = "api_token"

# The Brain listens on the LAN here (config.api_port default 8300). This is a
# non-secret default that pre-fills the config-flow form; the owner can override.
DEFAULT_BASE_URL = "http://jetson.local:8300"

PLATFORMS = [Platform.CONVERSATION]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Aerys Conversation from a config entry."""
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
