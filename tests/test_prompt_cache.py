"""Static system and assistant-history breakpoints on chat and action payloads.

The router and privacy judge remain uncached. Inspect real API payloads offline;
the dynamic current human must never acquire a breakpoint.
"""

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


def _markers(value):
    if isinstance(value, dict):
        return int("cache_control" in value) + sum(_markers(v) for v in value.values())
    if isinstance(value, list):
        return sum(_markers(v) for v in value)
    return 0


@pytest.mark.parametrize("armed", [False, True])
def test_chat_models_cache_static_system_and_last_assistant(armed):
    from langchain_core.messages import AIMessage
    from aerys_v2.factory import build_model, tier_models_for
    models = [build_model(settings(armed)), *tier_models_for(settings(armed)).values()]
    for model in models:
        if armed:
            model = model.primary
        payload = model._get_request_payload([
            SystemMessage(content="soul"), HumanMessage(content="old"), AIMessage(content="old reply"),
            HumanMessage(content="recent"), AIMessage(content="recent reply"),
            HumanMessage(content=[{"type": "text", "text": "dynamic"}, {"type": "text", "text": "now"}]),
        ])
        assert _markers(payload) == 2
        assert payload["system"][-1]["cache_control"] == EPHEMERAL
        assert payload["messages"][-2]["content"][-1]["cache_control"] == EPHEMERAL
        assert _markers(payload["messages"][-1]) == 0
        assert _markers(payload["messages"][:-2]) == 0


@pytest.mark.parametrize("content", ["reply", [{"type": "text", "text": "thinking aloud"},
                                               {"type": "tool_use", "id": "t", "name": "read", "input": {}}]])
def test_history_marker_on_last_assistant_and_idempotent(content):
    payload = {"system": "soul", "messages": [
        {"role": "assistant", "content": content}, {"role": "user", "content": "now"}]}
    for _ in range(5):
        mark_cached_prefix(payload)
        assert _markers(payload) == 2
        assert _markers(payload["messages"][-1]) == 0
        assert _markers(payload) <= 4
    assert payload["messages"][0]["content"][-1]["cache_control"] == EPHEMERAL


@pytest.mark.parametrize("eligible", [False, True])
def test_history_marker_skips_code_execution_blocks(eligible):
    blocks = ([{"type": "text", "text": "reply"}] if eligible else []) + [
        {"type": "server_tool_use", "name": "bash_code_execution", "id": "exec", "input": {}},
        {"type": "bash_code_execution_tool_result", "tool_use_id": "exec", "content": []},
    ]
    payload = mark_cached_prefix({"system": "soul", "messages": [
        {"role": "assistant", "content": blocks}, {"role": "user", "content": "now"}]})
    assert _markers(payload) == (2 if eligible else 1)
    assert _markers(payload["messages"][0]["content"][-2:]) == 0
    assert _markers(payload["messages"][-1]) == 0


def test_existing_breakpoints_are_normalized_without_mutating_source_blocks():
    from copy import deepcopy
    blocks = [{"type": "text", "text": str(i), "cache_control": EPHEMERAL} for i in range(5)]
    messages = [{"role": "assistant", "content": blocks}, {"role": "user", "content": blocks}]
    original = deepcopy(messages)
    payload = mark_cached_prefix({"system": blocks, "messages": messages,
                                  "tools": [{"name": "read", "cache_control": EPHEMERAL}]})
    assert _markers(payload) == 2
    assert _markers(payload["messages"][-1]) == 0
    assert messages == original


def test_history_marker_skips_tools_called_by_code_execution():
    payload = mark_cached_prefix({"system": "soul", "messages": [{"role": "assistant", "content": [
        {"type": "text", "text": "eligible"},
        {"type": "tool_use", "id": "c", "name": "read", "input": {},
         "caller": {"type": "code_execution_20250825", "tool_id": "exec"}},
    ]}, {"role": "user", "content": "now"}]})
    assert _markers(payload) == 2
    assert payload["messages"][0]["content"][0]["cache_control"] == EPHEMERAL
    assert "cache_control" not in payload["messages"][0]["content"][1]


@pytest.mark.parametrize("armed", [False, True])
def test_actual_router_and_privacy_judge_remain_uncached(monkeypatch, armed):
    import aerys_v2.anthropic_model as transport
    import aerys_v2.factory as factory
    from aerys_v2.router import router_for
    seen = []
    original = transport.build_metered_model
    def record(*args, **kwargs):
        model = original(*args, **kwargs)
        seen.append(model)
        return model
    monkeypatch.setattr(transport, "build_metered_model", record)
    monkeypatch.setattr(factory, "build_metered_model", record)
    router_for(settings(armed), "soul")
    factory.content_privacy_fn_for(settings(armed))
    assert len(seen) == 2
    for model in seen:
        payload = model._get_request_payload([SystemMessage(content="soul"), HumanMessage(content="hi")])
        assert _markers(payload) == 0
