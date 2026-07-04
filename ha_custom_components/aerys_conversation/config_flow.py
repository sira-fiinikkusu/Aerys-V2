"""Config flow for Aerys Conversation.

A single form asking for the Brain's base_url (default the LAN 8300 endpoint) and
the Bearer api_token. Keeping the token in the config entry — not in source —
means this component can be checked into the public repo with no secret in it.
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult

from . import CONF_API_TOKEN, CONF_BASE_URL, DEFAULT_BASE_URL, DOMAIN


class AerysConversationConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Aerys Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(title="Aerys", data=user_input)

        schema = vol.Schema(
            {
                vol.Required(CONF_BASE_URL, default=DEFAULT_BASE_URL): str,
                # The Brain's Bearer token (config.api_token) — entered on install,
                # stored in the config entry, never written into source.
                vol.Required(CONF_API_TOKEN): str,
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema)
