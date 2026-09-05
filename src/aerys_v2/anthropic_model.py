"""Metered transport policy when an offline lifeboat is armed."""

from functools import cached_property
from typing import Any

import anthropic
import httpx
from langchain_anthropic import ChatAnthropic

from aerys_v2.config import Settings


class _LifeboatPrimary(ChatAnthropic):
    """Keep generation budgets but abandon a black-holed connection in 5s.

    langchain-anthropic 1.4.6 only accepts float|None for `timeout`, and its
    _client_params compares that value with zero. Its HTTP-client factories also
    hash the timeout via lru_cache; httpx.Timeout (also anthropic.Timeout) is
    unhashable. Build the parent clients with the original scalar, then use the
    SDK's with_options to apply the split timeout to every request. This keeps
    the shared HTTP transports, headers, base URL and proxy behavior intact,
    without mutating another model's transport or passing Timeout to the cache.
    """

    @cached_property
    def _client(self) -> anthropic.Client:
        client = super()._client
        return client.with_options(timeout=httpx.Timeout(client.timeout, connect=5.0))

    @cached_property
    def _async_client(self) -> anthropic.AsyncClient:
        client = super()._async_client
        return client.with_options(timeout=httpx.Timeout(client.timeout, connect=5.0))


def build_metered_model(settings: Settings, **kwargs: Any) -> ChatAnthropic:
    """Preserve unarmed clients exactly; the lifeboat replaces SDK retries."""
    if getattr(settings, "local_fallback_url", None) is None:
        return ChatAnthropic(**kwargs)
    return _LifeboatPrimary(**{**kwargs, "max_retries": 0})
