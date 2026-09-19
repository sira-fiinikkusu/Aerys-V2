"""Metered transport policy when an offline lifeboat is armed."""

from copy import deepcopy
from functools import cached_property
from typing import Any

import anthropic
import httpx
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import (
    _apply_cache_control_to_last_eligible_block,
    _collect_code_execution_tool_ids,
)

from aerys_v2.config import Settings


EPHEMERAL = {"type": "ephemeral"}


def mark_cached_prefix(payload: dict) -> dict:
    """Cache tools + static system, then history through the last assistant.

    Measured 2026-09-19: the minute clock before history caused a cache write
    every turn, even 75s apart. Live context now trails history in the current
    human message; never mark that message. Keep the system breakpoint too:
    the action loop's forced→auto tool_choice switch invalidates message cache,
    but tools + system survive it. This policy owns at most two breakpoints.
    """
    payload.pop("cache_control", None)
    # Normalize incoming explicit markers so replayed blocks cannot accumulate
    # past Anthropic's four-breakpoint ceiling. Copy before touching any blocks.
    for key in ("system", "tools"):
        if isinstance(payload.get(key), list):
            payload[key] = [{k: v for k, v in block.items() if k != "cache_control"}
                            if isinstance(block, dict) else block for block in payload[key]]
    messages = [deepcopy(message) for message in payload.get("messages", [])]
    for message in messages:
        if isinstance(message.get("content"), list):
            message["content"] = [
                {k: v for k, v in block.items() if k != "cache_control"}
                if isinstance(block, dict) else block for block in message["content"]
            ]
    if "messages" in payload:
        payload["messages"] = messages
    system = payload.get("system")
    if isinstance(system, str) and system:
        payload["system"] = [{"type": "text", "text": system, "cache_control": EPHEMERAL}]
    elif isinstance(system, list) and system and isinstance(system[-1], dict):
        payload["system"] = [*system[:-1], {**system[-1], "cache_control": EPHEMERAL}]
    elif payload.get("tools"):
        payload["tools"] = [*payload["tools"][:-1], {**payload["tools"][-1], "cache_control": EPHEMERAL}]
    last_assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if last_assistant is not None:
        # 1.4.6's helper skips tool calls made by code_execution and their results.
        # Server-side execution blocks need the same exclusion.
        candidate = dict(last_assistant)
        if isinstance(candidate.get("content"), list):
            candidate["content"] = [
                block for block in candidate["content"]
                if not isinstance(block, dict) or not (
                    "code_execution" in block.get("type", "")
                    or (block.get("type") == "server_tool_use"
                        and "code_execution" in block.get("name", ""))
                )
            ]
        _apply_cache_control_to_last_eligible_block(
            [candidate], EPHEMERAL, _collect_code_execution_tool_ids(messages),
        )
        # List blocks are shared with our payload copy; string promotion isn't.
        if isinstance(last_assistant.get("content"), str):
            last_assistant["content"] = candidate["content"]
    return payload


class _PrefixCached(ChatAnthropic):
    """ChatAnthropic with system and last-assistant prompt-cache breakpoints.

    cache_prefix changes only the payload markers; transport, retries and
    timeouts are untouched.
    """

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

    cache_prefix=True asks for system and history prompt-cache breakpoints
    (see mark_cached_prefix); without it the plain ChatAnthropic is returned.
    """
    if getattr(settings, "local_fallback_url", None) is None:
        if kwargs.get("cache_prefix"):
            return _PrefixCached(**kwargs)
        return ChatAnthropic(**kwargs)
    return _LifeboatPrimary(**{**kwargs, "max_retries": 0})
