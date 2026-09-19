"""Prompt caching is pinned to the action path: every tool-turn model puts ONE
cache breakpoint on the SYSTEM block so the ~14k-token prefix (tools + overlay)
is read from cache on the second call of a turn. The breakpoint must NOT sit in
the messages: the first call forces a tool (tool_choice="any") and the second
does not, and a tool_choice change invalidates the message cache — measured
2026-09-19 as cache_write on every call and cache_read on none. The router and
the privacy judge carry no marker (router prompt is below Haiku's cache
minimum). Payloads are inspected offline via the real request builder."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from aerys_v2.anthropic_model import build_metered_model, mark_cached_prefix
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
        [SystemMessage(content="the action overlay"), HumanMessage(content="turn off the office")],
        **binding.kwargs,
    )


def _assert_system_breakpoint(payload):
    assert "cache_control" not in payload, "top-level marker would land in the messages"
    system = payload["system"]
    assert isinstance(system, list) and system[-1]["cache_control"] == EPHEMERAL
    assert system[-1]["text"] == "the action overlay"
    for message in payload["messages"]:
        for block in (message["content"] if isinstance(message["content"], list) else []):
            assert "cache_control" not in block


@pytest.mark.parametrize("armed", [False, True])
@pytest.mark.parametrize("pin", ["", "pinned"])
def test_every_tool_turn_model_sends_top_level_cache_control(armed, pin):
    pair = build_api_tool_model(settings(armed, action_model=pin), [test_light])
    for name in ("_conv", "_auto", "_forced", "_fast_auto", "_fast_forced"):
        payload = _payload(getattr(pair, name), armed)
        _assert_system_breakpoint(payload)
        # The cache marker must not displace the things that make it a tool turn.
        assert payload["tools"], name
        assert payload["max_tokens"] == 1024, name


@pytest.mark.parametrize("armed", [False, True])
def test_toolless_conversational_model_caches_too(armed):
    empty = build_api_tool_model(settings(armed), [])
    _assert_system_breakpoint(_payload(empty, armed))


def test_breakpoint_falls_back_to_the_last_tool_without_system_text():
    payload = mark_cached_prefix({"system": "", "tools": [{"name": "a"}, {"name": "b"}],
                                  "messages": [], "cache_control": EPHEMERAL})
    assert "cache_control" not in payload
    assert payload["tools"][-1] == {"name": "b", "cache_control": EPHEMERAL}
    assert payload["tools"][0] == {"name": "a"}


def test_breakpoint_moves_to_the_last_block_of_a_list_system():
    payload = mark_cached_prefix({"system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})
    assert payload["system"][0] == {"type": "text", "text": "a"}
    assert payload["system"][1] == {"type": "text", "text": "b", "cache_control": EPHEMERAL}


def test_cache_control_stays_off_the_router_and_judge_prompts():
    # Same builder, no model_kwargs: the router and judge call sites pass none,
    # so their payloads carry no top-level cache_control. A future "cache all
    # metered calls" refactor must change this test on purpose.
    model = build_metered_model(settings(False), model="claude-haiku-4-5", api_key="sk-test",
                                max_tokens=200, timeout=10.0, max_retries=1)
    payload = model._get_request_payload([SystemMessage(content="soul"), HumanMessage(content="hello")])
    assert "cache_control" not in payload
    assert payload["system"] == "soul"
