"""Prompt caching is pinned to the action path: every tool-turn model sends a
top-level cache_control so the ~14k-token prefix (overlay + tools) is read from
cache on the second call of a turn; the router and the privacy judge do NOT —
the router prompt is below Haiku's cache minimum, and the judge has no prefix
worth caching. Payloads are inspected offline via the real request builder."""

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from aerys_v2.anthropic_model import build_metered_model
from aerys_v2.config import Settings
from aerys_v2.factory import build_api_tool_model

EPHEMERAL = {"type": "ephemeral"}


@tool
def test_light() -> str:
    """Read the fake light."""
    return "off"


def settings(armed=True, **overrides):
    return Settings(
        _env_file=None, anthropic_api_key="sk-test",
        local_fallback_url="http://127.0.0.1:11434/v1" if armed else None,
        **overrides,
    )


def _payload(binding, armed):
    if armed:
        binding = binding.primary
    # bind_tools() keeps the tools in the binding's kwargs; merge them the way
    # invoke() does so the payload is the one the API would actually receive.
    return binding.bound._get_request_payload(
        [HumanMessage(content="turn off the office")], **binding.kwargs
    )


@pytest.mark.parametrize("armed", [False, True])
@pytest.mark.parametrize("pin", ["", "pinned"])
def test_every_tool_turn_model_sends_top_level_cache_control(armed, pin):
    pair = build_api_tool_model(settings(armed, action_model=pin), [test_light])
    for name in ("_conv", "_auto", "_forced", "_fast_auto", "_fast_forced"):
        payload = _payload(getattr(pair, name), armed)
        assert payload["cache_control"] == EPHEMERAL, name
        # The cache marker must not displace the things that make it a tool turn.
        assert payload["tools"], name
        assert payload["max_tokens"] == 1024, name


@pytest.mark.parametrize("armed", [False, True])
def test_toolless_conversational_model_caches_too(armed):
    empty = build_api_tool_model(settings(armed), [])
    assert _payload(empty, armed)["cache_control"] == EPHEMERAL


def test_cache_control_stays_off_the_router_and_judge_prompts():
    # Same builder, no model_kwargs: the router and judge call sites pass none,
    # so their payloads carry no top-level cache_control. A future "cache all
    # metered calls" refactor must change this test on purpose.
    model = build_metered_model(settings(False), model="claude-haiku-4-5", api_key="sk-test",
                                max_tokens=200, timeout=10.0, max_retries=1)
    assert "cache_control" not in model._get_request_payload([HumanMessage(content="hello")])
