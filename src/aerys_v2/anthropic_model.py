"""Metered transport policy when an offline lifeboat is armed."""

from functools import cached_property
from typing import Any

import anthropic
import httpx
from langchain_anthropic import ChatAnthropic

from aerys_v2.config import Settings


EPHEMERAL = {"type": "ephemeral"}


def mark_cached_prefix(payload: dict) -> dict:
    """Place ONE cache breakpoint on the system block (falling back to the last
    tool when there is no system text) and drop any top-level cache_control.

    Why the system block and not the API's automatic (last-block) placement:
    an action turn calls the model twice — first with tool_choice="any", then
    auto — and a tool_choice change invalidates the MESSAGE cache. A breakpoint
    inside the messages is therefore rewritten on the second call (measured
    2026-09-19: cache_write on every call, cache_read never). Cache prefixes
    are built tools → system → messages, so a breakpoint on the system block
    caches tools + system regardless of tool_choice, and survives the switch.
    """
    payload.pop("cache_control", None)
    system = payload.get("system")
    if isinstance(system, str) and system:
        payload["system"] = [{"type": "text", "text": system, "cache_control": EPHEMERAL}]
    elif isinstance(system, list) and system and isinstance(system[-1], dict):
        payload["system"] = [*system[:-1], {**system[-1], "cache_control": EPHEMERAL}]
    elif payload.get("tools"):
        payload["tools"] = [*payload["tools"][:-1], {**payload["tools"][-1], "cache_control": EPHEMERAL}]
    return payload


class _PrefixCached(ChatAnthropic):
    """ChatAnthropic whose request carries a prompt-cache breakpoint on the
    system block when cache_prefix is set. Transport, retries and timeouts are
    untouched; only the payload gains the marker."""

    cache_prefix: bool = False

    def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[override]
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        return mark_cached_prefix(payload) if self.cache_prefix else payload


class _LifeboatPrimary(_PrefixCached):
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
    """Preserve unarmed clients exactly; the lifeboat replaces SDK retries.

    cache_prefix=True asks for a prompt-cache breakpoint on the system block
    (see mark_cached_prefix); without it the plain ChatAnthropic is returned.
    """
    if getattr(settings, "local_fallback_url", None) is None:
        if kwargs.get("cache_prefix"):
            return _PrefixCached(**kwargs)
        return ChatAnthropic(**kwargs)
    return _LifeboatPrimary(**{**kwargs, "max_retries": 0})
