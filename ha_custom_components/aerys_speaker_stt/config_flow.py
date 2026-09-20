"""Config flow: which STT entity to wrap, where the recognizer lives, how long to wait."""
from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult

from .const import (
    CONF_RECOGNIZER_URL, CONF_SOURCE_STT, CONF_TIMEOUT, DEFAULT_RECOGNIZER_URL,
    DEFAULT_SOURCE_STT, DEFAULT_TIMEOUT, DOMAIN,
)


class AerysSpeakerSttConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            await self.async_set_unique_id(user_input[CONF_SOURCE_STT])
            self._abort_if_unique_id_configured()
            return self.async_create_entry(title=f"Aerys speaker ({user_input[CONF_SOURCE_STT]})", data=user_input)
        schema = vol.Schema({
            vol.Required(CONF_SOURCE_STT, default=DEFAULT_SOURCE_STT): str,
            vol.Required(CONF_RECOGNIZER_URL, default=DEFAULT_RECOGNIZER_URL): str,
            vol.Required(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): vol.Coerce(float),
        })
        return self.async_show_form(step_id="user", data_schema=schema)
